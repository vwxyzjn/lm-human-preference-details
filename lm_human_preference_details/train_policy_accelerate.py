import os
import random
import time
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace
from typing import Optional

from accelerate import Accelerator
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tyro
from rich.console import Console
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
    kl_coef: float = 0.25
    adaptive_kl: Optional[AdaptiveKLParams] = field(default_factory=AdaptiveKLParams)
    trained_model: Optional[str] = "models/reward.pt"
    label_dataset: tyro.conf.Suppress[Optional[str]] = None


@dataclass
class PpoHParams:
    total_episodes: int = 1000000
    local_batch_size: int = 64
    local_mini_batch_size: tyro.conf.Suppress[int] = None
    batch_size: tyro.conf.Suppress[int] = None
    mini_batch_size: tyro.conf.Suppress[int] = None
    world_size: tyro.conf.Suppress[int] = None
    batch_size: tyro.conf.Suppress[int] = None
    minibatch_size: tyro.conf.Suppress[int] = None
    num_updates: tyro.conf.Suppress[int] = None
    nminibatches: int = 1
    noptepochs: int = 4
    lr: float = 0.00001
    vf_coef: float = .1
    cliprange: float = .2
    cliprange_value: float = .2
    gamma: float = 1
    lam: float = 0.95
    whiten_rewards: bool = True


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
    exp_name: str = os.path.basename(__file__)[:-len(".py")]
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
    run_name: tyro.conf.Suppress[str] = None
    """TO BE FILLED: a unique name of this run"""

    base_model: str = "gpt2"
    """the name of the pretrained model to use"""
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
    torch.nn.init.normal_(layer.weight, std=std)
    torch.nn.init.constant_(layer.bias, val=bias_const)
    return layer

def whiten(values, shift_mean=True):
    # `unbiased=False` matches TF `tf.nn.moments`'s setting
    mean, var = torch.mean(values), torch.var(values, unbiased=False)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


OPENAI_PAD_TOKEN_ID = 50259


class ScalarHead(nn.Module):
    def __init__(self, config, scale=None, **kwargs):
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
        if scale is None:
            scale = 1 / np.sqrt(hidden_size + 1)
        self.summary = layer_init(nn.Linear(hidden_size, 1), std=scale)
        self.flatten = nn.Flatten()

    def forward(self, hidden_states):
        output = self.dropout(hidden_states)
        output = self.summary(output)
        return output


class AutoModelForCausalLMWithScalarHead(nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.scalar_head = ScalarHead(self.pretrained_model.config, scale=0.0)

    def forward(self, **kwargs):
        output = self.pretrained_model(**kwargs)
        return output, self.scalar_head(output.hidden_states[-1])


class AutoModelForCausalLMWithRewardHead(nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.scalar_head = ScalarHead(self.pretrained_model.config)
        self.reward_gain = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.reward_bias = torch.nn.Parameter(torch.tensor(0.0), requires_grad=True)


# a pytorch dataset
class MyDataset(IterableDataset):
    def __init__(self, generator, tokenizer, query_length, start_text=None, end_text=None, seed=None):
        self.generator = generator
        self.tokenizer = tokenizer
        self.query_length = query_length
        self.start_text = start_text
        self.end_text = end_text
        self.seed = seed
        token_to_index = tokenizer.get_vocab()
        self.start_token = token_to_index[start_text] if self.start_text else None
        self.end_token = token_to_index[end_text] if self.end_text else None

    def __iter__(self):
        for text in self.generator("train", self.seed, shuffle=True):
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


def left_padding_to_right_padding(query, pad_id):
    # got to convert to right padding, otherwise `transformers` has weird issues
    # even with `position_ids`
    return torch.tensor([
        [pad_id]*(row==pad_id).sum() + [x for x in row if x != pad_id]
        for row in query
    ])


def ceil_div(a, b):
    return (a - 1) // b + 1


def exact_div(a, b):
    q = a // b
    if a != q * b:
        raise ValueError('Inexact division: %s / %s = %s' % (a, b, a / b))
    return q


def generate(pretrained_model, queries, tokenizer, generation_config):
    """generate in a way that does not affect padding tokens"""
    context_length = queries.shape[1]
    attention_mask = queries != tokenizer.pad_token_id
    input_ids = queries.clone()
    input_ids[~attention_mask] = 0 # set padding tokens to 0
    output = pretrained_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        # position_ids=attention_mask.cumsum(1) - attention_mask.long(), # generation collapsed if this was turned on. TODO: why does generation collapse with this?
        generation_config=generation_config,
        return_dict_in_generate=True,
    )
    # restore padding tokens    
    return torch.cat((queries, output.sequences[:, context_length:]), dim=1)


def get_reward(reward_model, query_responses, tokenizer):
    attention_mask = query_responses != tokenizer.pad_token_id
    position_ids = attention_mask.cumsum(1) - attention_mask.long() # exclusive cumsum
    input_ids = query_responses.clone()
    input_ids[~attention_mask] = 0
    output = reward_model.pretrained_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        return_dict=True,
        output_hidden_states=True,
    )
    reward = reward_model.scalar_head(output.hidden_states[-1])
    reward = reward_model.reward_gain * reward + reward_model.reward_bias
    # but we only care about the reward of the last token
    reward = reward[:, -1]
    return reward

def forward(policy, query_responses, tokenizer):
    attention_mask = query_responses != tokenizer.pad_token_id
    position_ids = attention_mask.cumsum(1) - attention_mask.long() # exclusive cumsum
    input_ids = query_responses.clone()
    input_ids[~attention_mask] = 0
    return policy(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        return_dict=True,
        output_hidden_states=True,
    )

# if __name__ == "__main__":
#     args = tyro.cli(Args)
def train(args: Args):
    accelerator = Accelerator()
    args.ppo.world_size = accelerator.num_processes
    args.ppo.batch_size = int(args.ppo.local_batch_size * args.ppo.world_size)
    args.ppo.minibatch_size = exact_div(args.ppo.batch_size, args.ppo.nminibatches)

    args.ppo.local_mini_batch_size = exact_div(args.ppo.local_batch_size, args.ppo.nminibatches)
    if args.ppo.whiten_rewards:
        assert args.ppo.local_mini_batch_size >= 8, \
            f"Per-rank minibatch size {args.ppo.local_mini_batch_size} is insufficient for whitening"
    # `per_rank_rollout_batch_size` is our `args.ppo.local_batch_size`
    # `per_rank_minibatch_size` is our `args.ppo.local_mini_batch_size`
    args.ppo.num_updates = args.ppo.total_episodes // args.ppo.batch_size

    console = Console()
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SimpleNamespace() # dummy writer
    writer.add_scalar = lambda x, y, z: None
    if accelerator.is_main_process:
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
            wandb.run.log_code(".")
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        pprint(args)
    device = accelerator.device
    args.seed += accelerator.process_index
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        padding_side="right",
    )
    # we use the padding token manually but do not resize the token embedding of the model
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    reward_model = AutoModelForCausalLMWithRewardHead(AutoModelForCausalLM.from_pretrained(args.base_model)).to(device)
    if args.rewards.trained_model:
        reward_model.load_state_dict(torch.load(args.rewards.trained_model, map_location=device))
        print(f"loaded pretrained reward model from {args.rewards.trained_model}")
    # each class should have a sepatate pretrained model that do not share weights
    ref_policy = AutoModelForCausalLMWithScalarHead(AutoModelForCausalLM.from_pretrained(args.base_model)).to(device)
    policy = AutoModelForCausalLMWithScalarHead(AutoModelForCausalLM.from_pretrained(args.base_model)).to(device)
    policy.pretrained_model.generation_config.eos_token_id = None  # disable `pad_token_id` and `eos_token_id` because we just want to
    policy.pretrained_model.generation_config.pad_token_id = None  # generate tokens without truncation / padding
    # IMPORTANT: Layer norm produces weird gradients, which affects Adam optimizer to impact all the parameters systematically
    # In comparison, SGD does not appear to have this issue. TODO: add a link to the issue
    optimizer = optim.AdamW(policy.parameters(), eps=1e-5, betas=(0.9, 0.999))
    dataset = MyDataset(
        DATASET[args.task.query_dataset],
        tokenizer,
        args.task.query_length,
        start_text=args.task.start_text,
        end_text=args.task.end_text,
    )
    dataloader = DataLoader(dataset, batch_size=args.ppo.local_batch_size)
    policy, optimizer, dataloader = accelerator.prepare(policy, optimizer, dataloader)
    iter_dataloader = iter(dataloader)
    kl_ctl = AdaptiveKLController(args.rewards.kl_coef, hparams=args.rewards.adaptive_kl)
    # WARNING: even with `max_new_tokens` and `min_new_tokens` set to the same value, the number of tokens generated
    # may not be the same. TODO: investigate further, we just want to generate a fixed number of tokens
    generation_config = GenerationConfig(
        max_new_tokens=args.task.response_length,
        min_new_tokens=args.task.response_length,
        temperature=args.task.temperature,
        top_k=0.0,
        top_p=1.0,
        do_sample=True,
    )

    print("===training policy===")
    global_step = 0
    for update in range(1, args.ppo.num_updates + 1):
        global_step += 1 * args.ppo.batch_size
        frac = 1.0 - (update - 1.0) / args.ppo.num_updates
        lrnow = frac * args.ppo.lr
        optimizer.param_groups[0]["lr"] = lrnow
        data = next(iter_dataloader)
        with torch.no_grad():
            queries = data["input_ids"].to(device)
            queries = left_padding_to_right_padding(data["input_ids"], tokenizer.pad_token_id).to(device)
            query_responses = generate(accelerator.unwrap_model(policy).pretrained_model, queries, tokenizer, generation_config)
            context_length = queries.shape[1]
            responses = query_responses[:,context_length:]

            output, _ = forward(policy, query_responses, tokenizer)
            logits = output.logits[:,context_length-1:-1]
            logits /= args.task.temperature
            all_logprobs = F.log_softmax(logits, dim=-1)
            logprobs = torch.gather(all_logprobs, 2, responses.unsqueeze(-1)).squeeze(-1)
            values = accelerator.unwrap_model(policy).scalar_head(output.hidden_states[-1][:,context_length-1:-1]).squeeze(-1)

            ref_output, _ = forward(ref_policy, query_responses, tokenizer)
            ref_logits = ref_output.logits[:,context_length-1:-1]
            ref_logits /= args.task.temperature
            ref_all_logprobs = F.log_softmax(ref_logits, dim=-1)
            ref_logprobs = torch.gather(ref_all_logprobs, 2, responses.unsqueeze(-1)).squeeze(-1)

            # **Response Processing**
            # 1. truncate at the first occurrence of `truncate_token` that appears at or after
            # position truncate_after in the responses
            # https://github.com/openai/lm-human-preferences/blob/cbfd210bb8b08f6bc5c26878c10984b90f516c66/lm_human_preferences/train_policy.py#L378
            truncate_token_mask = (responses == args.task.truncate_token)
            truncate_after_or_token_mask = torch.cat([torch.zeros_like(truncate_token_mask)[:,:args.task.truncate_after], truncate_token_mask[:,args.task.truncate_after:]], dim=1)
            truncate_mask = (torch.cumsum(truncate_after_or_token_mask, dim=1) - truncate_after_or_token_mask.long()).bool()
            postprocessed_responses = torch.where(truncate_mask, torch.full_like(responses, tokenizer.pad_token_id), responses)

            # 2. run reward model on the truncated responses
            # TODO: fix position ids
            postprocessed_query_responses = torch.cat((queries, postprocessed_responses), 1)
            scores = get_reward(reward_model, postprocessed_query_responses, tokenizer).flatten()

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

            # 5. whiten rewards
            if args.ppo.whiten_rewards:
                rewards = whiten(rewards, shift_mean=False)
            try:
                sample_kl = kl[0].sum().item()
                postprocessed_responses = postprocessed_query_responses[:,context_length:]
                console.print(f"[green]{tokenizer.decode(queries[0], skip_special_tokens=True)}[/]\n[yellow]{tokenizer.decode(postprocessed_responses[0], skip_special_tokens=True)}[/]\n[blue](NO POST-PROCESSING){tokenizer.decode(responses[0], skip_special_tokens=True)}[/]\n[red]score: {scores[0]}, kl: {kl[0].sum().item()}, total reward: {scores[0] - kl_ctl.value * sample_kl} [/]")
            except:
                pass

        # Do multiple epochs of PPO training, with a fresh random shuffle in each epoch
        for ppo_epoch_idx in range(args.ppo.noptepochs):
            order = np.random.permutation(args.ppo.local_batch_size)
            for mb_start in range(0, args.ppo.local_batch_size, args.ppo.local_mini_batch_size):
                # The reference codebase does not use minibatch really but we should implement it
                # TODO: implmenet mini batch size
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

                output, vpred_temp = forward(policy, query_responses, tokenizer)
                logits = output.logits[:,context_length-1:-1]
                logits /= args.task.temperature
                new_all_logprobs = F.log_softmax(logits, dim=-1)
                new_logprobs = torch.gather(new_all_logprobs, 2, responses.unsqueeze(-1)).squeeze(-1)
                vpred = vpred_temp[:,context_length-1:-1].squeeze(-1)
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
                accelerator.backward(loss)
                grads = [
                    param.grad.detach().flatten()
                    for param in policy.parameters()
                    if param.grad is not None
                ]
                norm = torch.cat(grads).norm()
                optimizer.step()
                pd = torch.nn.functional.softmax(logits, dim=-1)
                entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
                approxkl = .5 * (logprobs_diff ** 2).mean()
                return_mean, return_var = returns.mean(), returns.var()
                value_mean, value_var = values.mean(), values.var()
            if accelerator.is_main_process:
                console.print(f"ppo_epoch_idx", ppo_epoch_idx, "approxkl", approxkl.item(), "pg_loss", pg_loss.item(), "pg_clipfrac", pg_clipfrac.item(), "ratio", ratio.mean().item())

        kl = logprobs - ref_logprobs
        mean_kl = kl.sum(1).mean()
        mean_entropy = (-logprobs).sum(1).mean()
        mean_non_score_reward = non_score_reward.sum(1).mean()
        writer.add_scalar("objective/kl_coef", kl_ctl.value, update)
        writer.add_scalar("objective/kl", accelerator.gather(mean_kl).mean().item(), update)
        writer.add_scalar("objective/entropy", accelerator.gather(mean_entropy).mean().item(), update)
        writer.add_scalar("objective/non_score_reward", accelerator.gather(mean_non_score_reward).mean().item(), update)
        writer.add_scalar("objective/score_total", accelerator.gather(mean_non_score_reward + scores.mean()).mean().item(), update)
        writer.add_scalar("objective/scores", accelerator.gather(scores.mean()).mean().item(), update)
        writer.add_scalar("ppo/loss/policy", accelerator.gather(pg_loss).mean().item(), update)
        writer.add_scalar("ppo/loss/value", accelerator.gather(vf_loss).mean().item(), update)
        writer.add_scalar("ppo/loss/total", accelerator.gather(loss).mean().item(), update)
        writer.add_scalar("ppo/loss/norm", accelerator.gather(norm).mean().item(), update)
        writer.add_scalar("ppo/policy/entropy", accelerator.gather(entropy.mean()).mean().item(), update)
        writer.add_scalar("ppo/policy/approxkl", accelerator.gather(approxkl).mean().item(), update)
        writer.add_scalar("ppo/policy/clipfrac", accelerator.gather(pg_clipfrac).mean().item(), update)
        writer.add_scalar("ppo/returns/mean", accelerator.gather(return_mean).mean().item(), update)
        writer.add_scalar("ppo/returns/var", accelerator.gather(return_var).mean().item(), update)
        writer.add_scalar("ppo/val/vpred", accelerator.gather(vpred.mean()).mean().item(), update)
        writer.add_scalar("ppo/val/error", accelerator.gather(vf_losses1.mean()).mean().item(), update)
        writer.add_scalar("ppo/val/clipfrac", accelerator.gather(vf_clipfrac).mean().item(), update)
        writer.add_scalar("ppo/val/mean", accelerator.gather(value_mean).mean().item(), update)
        writer.add_scalar("ppo/val/var", accelerator.gather(value_var).mean().item(), update)
        writer.add_scalar("ppo/val/ratio", accelerator.gather(ratio.mean()).mean().item(), update)
        writer.add_scalar("ppo/val/ratio_var", accelerator.gather(ratio.mean()).var().item(), update)
        writer.add_scalar("ppo/val/advantage", accelerator.gather(advantages.mean()).mean().item(), update)
        writer.add_scalar("ppo/val/advantage_var", accelerator.gather(advantages.mean()).var().item(), update)
        writer.add_scalar("ppo/val/num_eos_tokens", (responses == tokenizer.eos_token_id).sum().item(), update)
        writer.add_scalar("ppo/lr", lrnow, update)
        writer.add_scalar("ppo/episode", global_step, update)
        kl_ctl.update(mean_kl.item(), args.ppo.batch_size)

    # save model
    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        torch.save(reward_model.state_dict(), args.save_path)

if __name__ == "__main__":
    args = tyro.cli(Args)
    train(args)
