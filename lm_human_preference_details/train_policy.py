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
from transformers import AutoModelForCausalLM, AutoTokenizer

from lm_human_preference_details.data import DATASET


@dataclass
class AdaptiveKLParams:
    target: float = None
    horizon: int = 10000  # in episodes


@dataclass
class RewardHParams:
    kl_coef: float = 0.2
    adaptive_kl: Optional[AdaptiveKLParams] = None

    trained_model: Optional[str] = "models/reward.pt"

    # def validate(self, *, prefix=''):
    #     super().validate(prefix=prefix)
    #     assert self.trained_model is None or self.train_new_model is None, 'Cannot use trained_model and train new model'
    #     assert self.trained_model is not None or self.train_new_model is not None, 'Need either trained_model or to train a new model'


@dataclass
class PpoHParams:
    total_episodes: int = 1000000
    local_batch_size: int = 512
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
    truncate_token: Optional[int] = None
    truncate_after: int = 0
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


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.normal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


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
    # reward_model = AutoModelForCausalLMWithRewardHead(pretrained_model).to(device)
    # if args.rewards.trained_model:
    #     reward_model.load_state_dict(torch.load(args.rewards.trained_model))
    #     print("loaded pretrained reward model")
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

    print("===training policy===")
    global_step = 0
    num_updates = args.ppo.total_episodes // args.ppo.batch_size
    for update in range(1, num_updates + 1):
        global_step += 1 * args.ppo.batch_size

        responses = []
        logprobs = []
        values = []
        
        data = next(iter_dataloader)
        queries = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        output = policy.pretrained_model.generate(
            input_ids=queries,
            attention_mask=attention_mask,
            max_new_tokens=args.task.response_length,
            temperature=1.0, # TODO: change it back to args.task.temperature
            pad_token_id=tokenizer.pad_token_id, # ? should I do this?
            return_dict_in_generate=True,
            output_hidden_states=True,
            output_scores=True,
        )
        # extract logprobs and values from `output.hidden_states`
        responses = torch.stack([sequence[len(queries[0]):] for sequence in output.sequences])
        all_logprobs = torch.stack(output.scores, 1)
        # responses have shape (batch_size, response_length)
        # all_logprobs have shape (batch_size, response_length, vocab_size)
        logprobs = torch.gather(all_logprobs, 2, responses.unsqueeze(-1)).squeeze(-1)
        # len(output.hidden_states) = args.task.response_length
        # len(output.hidden_states[-1]) = num of layers
        # value is computed based on the last layer and last token
        values = policy.scalar_head(output.hidden_states[-1][-1]).squeeze(-1)

        # the attention mask should be the same as the one used in the policy
        assert ((output.sequences != tokenizer.pad_token_id)[:,:len(queries[0])] == attention_mask).all()
        with torch.no_grad():
            ref_output = ref_policy.pretrained_model(
                input_ids=output.sequences,
                return_dict=True,
                output_hidden_states=True,
            )
            # NOTE: len(ref_output.hidden_states) = num of layers, which is a different behavior from `output_hidden_states` in `generate`
            ref_logits = ref_output.logits[:,len(queries[0]):]
            ref_logits_calculated = ref_policy.pretrained_model.lm_head(ref_output.hidden_states[-1][:,len(queries[0]):])
            assert (ref_logits_calculated != ref_logits).sum() == 0
            ref_all_logprobs = F.log_softmax(ref_logits, dim=-1)
            ref_logprobs = torch.gather(ref_all_logprobs, 2, responses.unsqueeze(-1)).squeeze(-1)

        # ref_logprobs = ref_policy.analyze_responses(queries, responses)['logprobs']
        # scores, postprocessed_responses, score_stats = self.score_fn(queries, responses)
        raise
        # sample_queries_responses.append(query_response)
        # def respond_op(self, queries, length):
        #     contexts = self.embed_queries(queries)
        #     context_length = tf.shape(contexts)[1]
        #     result = sample.sample_sequence(
        #         step=self.step_core,
        #         context=contexts,
        #         length=length,
        #         model_hparams=self.model_hparams,
        #         temperature=self.temperature,
        #         extra_outputs={'values':tf.float32},
        #     )
        #     return dict(
        #         responses=result['tokens'][:, context_length:],
        #         logprobs=result['logprobs'],
        #         values=result['values'],
        #     )
        # rollouts = self.policy.respond(queries, length=args.task.response_length)

        # responses = rollouts['responses']
        # logprobs = rollouts['logprobs']
        # rollouts['queries'] = queries
        # ref_logprobs = self.ref_policy.analyze_responses(queries, responses)['logprobs']
        # scores, postprocessed_responses, score_stats = self.score_fn(queries, responses)

        # rewards, non_score_reward, kl_coef = self.compute_rewards(
        #     scores=scores,
        #     logprobs=logprobs,
        #     ref_logprobs=ref_logprobs)
        # rollouts['rewards'] = rewards

        # train_stats = self.train(rollouts=rollouts)

        # _, stats = self.record_step_stats(
        #     scores=scores, logprobs=logprobs, ref_logprobs=ref_logprobs, non_score_reward=non_score_reward,
        #     train_stats=train_stats, score_stats=score_stats, kl_coef=kl_coef)

        # self.kl_ctl.update(stats['objective/kl'], args.ppo.batch_size)

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
