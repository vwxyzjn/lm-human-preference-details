import os
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tyro
from datasets import load_dataset
from rich.pretty import pprint
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from lm_human_preference_details.data import DATASET


@dataclass
class AdaptiveKLParams:
    target: float = 6.0
    horizon: int = 10000  # in episodes


@dataclass
class RewardHParams:
    kl_coef: float = 0.2
    adaptive_kl: Optional[AdaptiveKLParams] = field(default_factory=AdaptiveKLParams)

    trained_model: Optional[str] = "models/reward.pt"

    # def validate(self, *, prefix=''):
    #     super().validate(prefix=prefix)
    #     assert self.trained_model is None or self.train_new_model is None, 'Cannot use trained_model and train new model'
    #     assert self.trained_model is not None or self.train_new_model is not None, 'Need either trained_model or to train a new model'


@dataclass
class PpoHParams:
    total_episodes: int = 1000000
    local_batch_size: int = int(512 / 8)
    local_mini_batch_size: tyro.conf.Suppress[int] = None
    batch_size: tyro.conf.Suppress[int] = None
    mini_batch_size: tyro.conf.Suppress[int] = None
    nminibatches: int = 1
    noptepochs: int = 4
    lr: float = 5e-6
    vf_coef: float = .1
    cliprange: float = .2
    cliprange_value: float = .2
    gamma: float = 1
    lam: float = 0.95
    whiten_rewards: bool = True

@dataclass
class LabelHParams:
    type: str = None
    num_train: int = 4992
    num_labels: int = 4
    source: str = None


@dataclass
class TaskHParams:
    # Query params
    query_length: int = 64
    query_dataset: str = "books"
    query_prefix: str = ""
    query_suffix: str = ""
    start_text: Optional[str] = None
    end_text: Optional[str] = None

    # Response params
    response_length: int = 24

    # Truncate response after the first occurrence of this token at or after index after when sampling.
    truncate_token: int = 13
    truncate_after: int = 16
    penalty_reward_value: int = -1

    # LM params
    temperature: float = 0.7


@dataclass
class Args:
    # common args
    exp_name: str = os.path.basename(__file__).rstrip(".py")
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanrl"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    cuda: bool = True
    """Whether to use cuda if available."""

    base_model: str = "gpt2"
    """the name of the pretrained model to use"""
    label_dataset: str = "sentiment/offline_5k.json"
    """the name of the dataset to use for labels in `https://huggingface.co/datasets/vwxyzjn/lm-human-preferences`"""
    local_batch_size: int = 32
    """per rank batch size"""
    lr: float = 0.00005
    """the learning rate"""
    local_rollout_batch_size: int = 128
    """per rank rollot batch size"""
    normalize_samples: int = 256
    """Samples used to estimate reward mean and std"""
    debug_normalize: int = 0
    """Samples used to check that normalization worked"""
    normalize_before: bool = True
    """Whether, before training, to normalize the rewards on the policy to the scales on the training buffer. (For comparisons, just use mean 0, var 1.)"""
    normalize_after: bool = True
    """Whether, after training, to normalize the rewards on the ref policy to mean 0, var 1 (so the KL coefficient always has the same meaning)."""
    print_sample_output_freq: int = 0
    """How often to print sample output"""
    save_path: str = "models/policy.pt"
    """Where to save the model"""

    task: TaskHParams = field(default_factory=TaskHParams)
    rewards: RewardHParams = field(default_factory=RewardHParams)
    ppo: PpoHParams = field(default_factory=PpoHParams)


class AdaptiveKLController:
    def __init__(self, init_kl_coef: float, hparams: AdaptiveKLParams):
        self.value = init_kl_coef
        self.hparams = hparams

    def update(self, current, n_steps):
        target = self.hparams.target
        proportional_error = np.clip(current / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.hparams.horizon
        self.value *= mult


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.normal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def whiten(values, shift_mean=True):
    mean, var = torch.mean(values), torch.var(values)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


OPENAI_PAD_TOKEN_ID = 50259


class ScalarHead(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        if not hasattr(config, "summary_dropout_prob"):
            summary_dropout_prob = kwargs.pop("summary_dropout_prob", 0.1)
        else:
            summary_dropout_prob = config.summary_dropout_prob
        self.dropout = nn.Dropout(summary_dropout_prob) if summary_dropout_prob else nn.Identity()
        # some models such as OPT have a projection layer before the word embeddings - e.g. OPT-350m
        if hasattr(config, "word_embed_proj_dim"):
            hidden_size = config.word_embed_proj_dim
        else:
            hidden_size = config.hidden_size
        self.summary = layer_init(nn.Linear(hidden_size, 1), std=1 / np.sqrt(hidden_size + 1))
        self.flatten = nn.Flatten()

    def forward(self, hidden_states):
        output = self.dropout(hidden_states)
        output = self.summary(output)
        return output




class AutoModelForCausalLMWithScalarHead(nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.scalar_head = ScalarHead(self.pretrained_model.config)

    def forward(self, input_ids, attention_mask=None):
        output = self.pretrained_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        reward = self.scalar_head(output.hidden_states[-1])
        return reward


class AutoModelForCausalLMWithRewardHead(AutoModelForCausalLMWithScalarHead):
    def __init__(self, pretrained_model):
        super().__init__(pretrained_model)
        self.reward_gain = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.reward_bias = torch.nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def forward(self, input_ids, attention_mask=None):
        output = self.pretrained_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        reward = self.scalar_head(output.hidden_states[-1])
        reward = self.reward_gain * reward + self.reward_bias
        return reward


# a pytorch dataset
class MyDataset(IterableDataset):
    def __init__(self, generator, tokenizer, query_length, start_text=None, end_text=None):
        self.generator = generator
        self.tokenizer = tokenizer
        self.query_length = query_length
        self.start_text = start_text
        self.end_text = end_text
        token_to_index = tokenizer.get_vocab()
        self.start_token = token_to_index[start_text] if self.start_text else None
        self.end_token = token_to_index[end_text] if self.end_text else None

    def __iter__(self):
        for text in self.generator("train", args.seed):
            tokens = self.tokenizer.encode(text)
            if self.start_token is not None:
                try:
                    first_index = tokens.index(self.start_token) + 1
                    if first_index < len(tokens):
                        tokens = tokens[first_index:]
                except:
                    continue
            tokens = tokens[: self.query_length]
            if self.end_token is not None:
                try:
                    last_index = len(tokens) - tokens[::-1].index(self.end_token)
                    tokens = tokens[:last_index]
                except:
                    continue
            output = self.tokenizer.pad(
                {"input_ids": tokens},
                padding="max_length",
                max_length=self.query_length,
                return_tensors="pt",
                return_attention_mask=True,
            )
            yield output


def ceil_div(a, b):
    return (a - 1) // b + 1


def exact_div(a, b):
    q = a // b
    if a != q * b:
        raise ValueError('Inexact division: %s / %s = %s' % (a, b, a / b))
    return q


if __name__ == "__main__":
    args = tyro.cli(Args)
    # calculation
    args.ppo.local_mini_batch_size = exact_div(args.ppo.local_batch_size, args.ppo.nminibatches)
    if args.ppo.whiten_rewards:
        assert args.ppo.local_mini_batch_size >= 8, \
            f"Per-rank minibatch size {args.ppo.local_mini_batch_size} is insufficient for whitening"
    args.ppo.batch_size = args.ppo.local_batch_size # * NUM_RANKS
    args.ppo.mini_batch_size = args.ppo.local_mini_batch_size # * NUM_RANKS
    # `per_rank_rollout_batch_size` is our `args.ppo.local_batch_size`
    # `per_rank_minibatch_size` is our `args.ppo.local_mini_batch_size`

    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=asdict(args),
            name=run_name,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    pprint(args)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        padding_side="right",
    )
    tokenizer.pad_token = tokenizer.eos_token
    pretrained_model = AutoModelForCausalLM.from_pretrained(args.base_model).to(device)
    reward_model = AutoModelForCausalLMWithRewardHead(pretrained_model).to(device)
    if args.rewards.trained_model:
        reward_model.load_state_dict(torch.load(args.rewards.trained_model))
        print("loaded pretrained reward model")
    # each class should have a sepatate pretrained model that do not share weights
    ref_policy = AutoModelForCausalLMWithScalarHead(AutoModelForCausalLM.from_pretrained(args.base_model)).to(device)
    policy = AutoModelForCausalLMWithScalarHead(AutoModelForCausalLM.from_pretrained(args.base_model)).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)
    dataset = MyDataset(
        DATASET[args.task.query_dataset],
        tokenizer,
        args.task.query_length,
        start_text=args.task.start_text,
        end_text=args.task.end_text,
    )
    dataloader = DataLoader(dataset, batch_size=args.ppo.local_batch_size)
    iter_dataloader = iter(dataloader)
    kl_ctl = AdaptiveKLController(args.rewards.kl_coef, hparams=args.rewards.adaptive_kl)
    generation_config = GenerationConfig(
        max_new_tokens=args.task.response_length,
        temperature=args.task.temperature,
        top_k=0.0,
        top_p=1.0,
        do_sample=True,
    )


    print("===training policy===")
    global_step = 0
    num_updates = args.ppo.total_episodes // args.ppo.batch_size
    for update in range(1, num_updates + 1):
        global_step += 1 * args.ppo.batch_size
        frac = 1.0 - (update - 1.0) / num_updates
        lrnow = frac * args.ppo.lr
        optimizer.param_groups[0]["lr"] = lrnow


        
        data = next(iter_dataloader)
        queries = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        with torch.no_grad():
            # TODO: this can be inefficient; if we implement our own `generate` we can generate the logprobs and values at the same time
            output = policy.pretrained_model.generate(
                input_ids=queries,
                attention_mask=attention_mask,
                generation_config=generation_config,
                pad_token_id=tokenizer.pad_token_id, # ? should I do this?
                return_dict_in_generate=True,
                output_hidden_states=True,
                output_scores=True,
            )
            query_responses = output.sequences
            responses = query_responses[:,len(queries[0]):]

            output = policy.pretrained_model(
                input_ids=query_responses,
                attention_mask=(query_responses != tokenizer.pad_token_id),
                return_dict=True,
                output_hidden_states=True,
            )
            logits = output.logits[:,len(queries[0]):]
            all_logprobs = F.log_softmax(logits, dim=-1)
            logprobs = torch.gather(all_logprobs, 2, responses.unsqueeze(-1)).squeeze(-1)
            values = policy.scalar_head(output.hidden_states[-1])[:,len(queries[0]):].squeeze(-1)

            ref_output = ref_policy.pretrained_model(
                input_ids=query_responses,
                attention_mask=(query_responses != tokenizer.pad_token_id),
                return_dict=True,
                output_hidden_states=True,
            )
            # NOTE: len(ref_output.hidden_states) = num of layers, which is a different behavior from `output_hidden_states` in `generate`
            ref_logits = ref_output.logits[:,len(queries[0]):]
            # ref_logits_calculated = ref_policy.pretrained_model.lm_head(ref_output.hidden_states[-1][:,len(queries[0]):])
            # assert (ref_logits_calculated != ref_logits).sum() == 0
            ref_all_logprobs = F.log_softmax(ref_logits, dim=-1)
            ref_logprobs = torch.gather(ref_all_logprobs, 2, responses.unsqueeze(-1)).squeeze(-1)

        # torch.testing.assert_close(logprobs, ref_logprobs)


        # response processing
        # 1. truncate at the first occurrence of `truncate_token` that appears at or after
        # position truncate_after in the responses
        # https://github.com/openai/lm-human-preferences/blob/cbfd210bb8b08f6bc5c26878c10984b90f516c66/lm_human_preferences/train_policy.py#L378
        truncate_token_mask = (responses == args.task.truncate_token)
        truncate_after_or_token_mask = torch.cat([torch.zeros_like(truncate_token_mask)[:,:args.task.truncate_after], truncate_token_mask[:,args.task.truncate_after:]], dim=1)
        truncate_mask = (torch.cumsum(truncate_after_or_token_mask, dim=1) - truncate_after_or_token_mask.long()).bool()
        postprocessed_responses = torch.where(truncate_mask, torch.full_like(responses, tokenizer.pad_token_id), responses)

        # 2. run reward model on the truncated responses
        postprocessed_query_responses = torch.cat((queries, postprocessed_responses), 1)
        scores = reward_model(
            postprocessed_query_responses,
            attention_mask=(postprocessed_query_responses != tokenizer.pad_token_id),
        )[:, -1].flatten()

        # 3. filter response. Ensure that the sample contains truncate_token
        # responses not passing that filter will receive a low (fixed) score
        # only query humans on responses that pass that filter
        matches_token = (postprocessed_responses[:, args.task.truncate_after:] == args.task.truncate_token)
        filter_mask = torch.any(matches_token, dim=-1)
        scores = torch.where(filter_mask, scores, torch.full_like(scores, args.task.penalty_reward_value))


        # 4. compute rewards
        kl = logprobs - ref_logprobs
        non_score_reward = -kl_ctl.value * kl
        rewards = non_score_reward.clone()
        rewards[:, -1] += scores

        stat_list = []
        # Do multiple epochs of PPO training, with a fresh random shuffle in each epoch
        for ppo_epoch_idx in range(args.ppo.noptepochs):
            order = np.random.permutation(args.ppo.local_batch_size)
            for mb_start in range(0, args.ppo.local_batch_size, args.ppo.local_mini_batch_size):
                # mb_data = {k: v[order[mb_start:mb_start+args.ppo.local_mini_batch_size]]
                #             for k, v in rollouts.items()}
                # TODO: implmenet mini batch size



                if args.ppo.whiten_rewards:
                    rewards = whiten(rewards, shift_mean=False)

                with torch.no_grad():
                    lastgaelam = 0
                    advantages_reversed = []
                    gen_length = args.task.response_length
                    for t in reversed(range(gen_length)):
                        nextvalues = values[:, t + 1] if t < gen_length - 1 else 0.0
                        delta = rewards[:, t] + args.ppo.gamma * nextvalues - values[:, t]
                        lastgaelam = delta + args.ppo.gamma * args.ppo.lam * lastgaelam
                        advantages_reversed.append(lastgaelam)
                    advantages = torch.stack(advantages_reversed[::-1], axis=1)
                    returns = advantages + values
                    advantages = whiten(advantages)

                # outputs = self.policy.analyze_responses_op(rollouts['queries'], rollouts['responses'])

                output = policy.pretrained_model(
                    input_ids=query_responses,
                    attention_mask=(query_responses != tokenizer.pad_token_id),
                    return_dict=True,
                    output_hidden_states=True,
                )
                logits = output.logits[:,len(queries[0]):]
                new_all_logprobs = F.log_softmax(logits, dim=-1)
                new_logprobs = torch.gather(new_all_logprobs, 2, responses.unsqueeze(-1)).squeeze(-1)
                vpred = policy.scalar_head(output.hidden_states[-1])[:,len(queries[0]):].squeeze(-1)

                vpredclipped = torch.clamp(vpred, values - args.ppo.cliprange_value, values + args.ppo.cliprange_value)
                vf_losses1 = torch.square(vpred - returns)
                vf_losses2 = torch.square(vpredclipped - returns)
                vf_loss = 0.5 * torch.max(vf_losses1, vf_losses2).mean()
                vf_clipfrac = (vf_losses2 > vf_losses1).float().mean()

                logprobs_diff = new_logprobs - logprobs
                ratio = torch.exp(logprobs_diff)
                pg_losses = -advantages * ratio
                pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - args.ppo.cliprange, 1.0 + args.ppo.cliprange)
                
                pg_loss = torch.max(pg_losses, pg_losses2).mean()
                pg_clipfrac = (pg_losses2 > pg_losses).float().mean()

                loss = pg_loss + args.ppo.vf_coef * vf_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # def entropy_from_logits(logits):
                #     pd = tf.nn.softmax(logits, axis=-1)
                #     return tf.math.reduce_logsumexp(logits, axis=-1) - tf.reduce_sum(pd*logits, axis=-1)
                # translate above to pytorch
                pd = torch.nn.functional.softmax(logits, dim=-1)
                entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
                approxkl = .5 * (logprobs_diff ** 2).mean()

                return_mean, return_var = returns.mean(), returns.var()
                value_mean, value_var = values.mean(), values.var()


                # loss, utils.flatten_dict(stats, sep='/')


                # ppo_loss, stats = self.loss(rollouts)
                # ppo_train_op = utils.minimize(
                #     loss=ppo_loss, lr=lrnow, params=policy.get_params(), name='ppo_opt', comm=self.comm)
                # return ppo_train_op, stats
        
        # def record_step_stats(*, kl_coef, **data):
        #     ppo_summary_writer = utils.get_summary_writer(self.hparams.run.save_dir, subdir='ppo', comm=self.comm)

        #     kl = data['logprobs'] - data['ref_logprobs']
        #     mean_kl = tf.reduce_mean(tf.reduce_sum(kl, axis=1))
        #     mean_entropy = tf.reduce_mean(tf.reduce_sum(-data['logprobs'], axis=1))
        #     mean_non_score_reward = tf.reduce_mean(tf.reduce_sum(data['non_score_reward'], axis=1))
        #     stats = {
        #         'objective/kl': mean_kl,
        #         'objective/kl_coef': kl_coef,
        #         'objective/entropy': mean_entropy,
        #     }
        #     for k, v in data['train_stats'].items():
        #         stats[f'ppo/{k}'] = tf.reduce_mean(v, axis=0)
        #     for k, v in data['score_stats'].items():
        #         mean = tf.reduce_mean(v, axis=0)
        #         stats[f'objective/{k}'] = mean
        #         stats[f'objective/{k}_total'] = mean + mean_non_score_reward

        #     stats = utils.FlatStats.from_dict(stats).map_flat(
        #         partial(utils.mpi_allreduce_mean, comm=self.comm)).as_dict()

        #     # Add more statistics
        #     step = tf.train.get_global_step().read_value()
        #     stats['ppo/val/var_explained'] = 1 - stats['ppo/val/error'] / stats['ppo/returns/var']
        #     steps = step + 1
        #     stats.update({
        #         'elapsed/updates': steps,
        #         'elapsed/steps/serial': steps * hparams.task.response_length,
        #         'elapsed/steps/total': steps * hparams.ppo.batch_size * hparams.task.response_length,
        #         'elapsed/episodes': steps * hparams.ppo.batch_size,
        #     })

        #     # Time statistics
        #     total, delta = tf_times()
        #     stats.update({
        #         'elapsed/fps': tf.cast(hparams.ppo.batch_size * hparams.task.response_length / delta, tf.int32),
        #         'elapsed/time': total,
        #     })
        #     if ppo_summary_writer:
        #         record_op = utils.record_stats(
        #             stats=stats, summary_writer=ppo_summary_writer, step=step, log_interval=hparams.run.log_interval, name='ppo_stats', comm=self.comm)
        #     else:
        #         record_op = tf.no_op()
        #     return record_op, stats


        kl = logprobs - ref_logprobs
        mean_kl = kl.sum(1).mean()
        mean_entropy = (-logprobs).sum(1).mean()
        mean_non_score_reward = non_score_reward.sum(1).mean()
        stats = {
            'objective/kl': mean_kl,
            'objective/kl_coef': kl_ctl.value,
            'objective/entropy': mean_entropy,
        }
        writer.add_scalar("objective/kl", mean_kl.item(), update)
        writer.add_scalar("objective/kl_coef", kl_ctl.value, update)
        writer.add_scalar("objective/entropy", mean_entropy.item(), update)
        writer.add_scalar("objective/non_score_reward", mean_non_score_reward.item(), update)
        writer.add_scalar("objective/scores", scores.mean().item(), update)

        writer.add_scalar("ppo/loss/policy", pg_loss.item(), update)
        writer.add_scalar("ppo/loss/value", vf_loss.item(), update)
        writer.add_scalar("ppo/loss/total", loss.item(), update)
        writer.add_scalar("ppo/policy/entropy", entropy.mean().item(), update)
        writer.add_scalar("ppo/policy/approxkl", approxkl.item(), update)
        writer.add_scalar("ppo/policy/clipfrac", pg_clipfrac.item(), update)
        writer.add_scalar("ppo/returns/mean", return_mean.item(), update)
        writer.add_scalar("ppo/returns/var", return_var.item(), update)
        writer.add_scalar("ppo/val/vpred", vpred.mean().item(), update)
        writer.add_scalar("ppo/val/error", vf_losses1.mean().item(), update)
        writer.add_scalar("ppo/val/clipfrac", vf_clipfrac.item(), update)
        writer.add_scalar("ppo/val/mean", value_mean.item(), update)
        writer.add_scalar("ppo/val/var", value_var.item(), update)
        

        kl_ctl.update(mean_kl.item(), args.ppo.batch_size)
        # TODO: metrics: scores, rewards,

        # train_stats = self.train(rollouts=rollouts)

        # _, stats = self.record_step_stats(
        #     scores=scores, logprobs=logprobs, ref_logprobs=ref_logprobs, non_score_reward=non_score_reward,
        #     train_stats=train_stats, score_stats=score_stats, kl_coef=kl_coef)


        # self.print_samples(queries=queries, responses=postprocessed_responses,
        #                    scores=scores, logprobs=logprobs, ref_logprobs=ref_logprobs)

        # # Record profiles of the step times
        # step = tf.get_default_session().run(tf.train.get_global_step())
        # step_time = time.time() - step_started_at
        # eps_per_second = float(args.ppo.batch_size) / step_time
        # if self.comm.Get_rank() == 0:
        #     print(f"[ppo_step {step}] step_time={step_time:.2f}s, "
        #           f"eps/s={eps_per_second:.2f}")


    # save model
    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        torch.save(reward_model.state_dict(), args.save_path)
