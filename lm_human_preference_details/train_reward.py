import os
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from datasets import load_dataset
from rich.pretty import pprint
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from lm_human_preference_details.data import DATASET


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
    save_path: str = "models/reward.pt"
    """Where to save the model"""
    task: TaskHParams = field(default_factory=TaskHParams)
    labels: LabelHParams = field(default_factory=LabelHParams)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.normal_(layer.weight, std=std)
    torch.nn.init.constant_(layer.bias, val=bias_const)
    return layer


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

    def forward(self, input_ids, attention_mask=None):
        output = self.pretrained_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        reward = self.scalar_head(output.hidden_states[-1])
        return reward


class AutoModelForCausalLMWithRewardHead(nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.scalar_head = ScalarHead(self.pretrained_model.config)
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


def left_padding_to_right_padding(query, pad_id):
    # got to convert to right padding, otherwise `transformers` has weird issues
    # even with `position_ids`
    return torch.tensor([
        [pad_id]*(row==pad_id).sum() + [x for x in row if x != pad_id]
        for row in query
    ])


def ceil_div(a, b):
    return (a - 1) // b + 1


if __name__ == "__main__":
    args = tyro.cli(Args)
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
    reward_model = AutoModelForCausalLMWithRewardHead(AutoModelForCausalLM.from_pretrained(args.base_model)).to(device)
    optimizer = optim.Adam(reward_model.parameters(), lr=args.lr, eps=1e-5)
    dataset = MyDataset(
        DATASET[args.task.query_dataset],
        tokenizer,
        args.task.query_length,
        start_text=args.task.start_text,
        end_text=args.task.end_text,
    )
    dataloader = DataLoader(dataset, batch_size=args.local_rollout_batch_size)
    iter_dataloader = iter(dataloader)

    generation_config = GenerationConfig(
        max_new_tokens=args.task.response_length,
        min_new_tokens=args.task.response_length,
        temperature=args.task.temperature,
        top_k=0.0,
        top_p=1.0,
        do_sample=True,
    )

    # `label` has keys `['sample0', 'query', 'best', 'sample3', 'sample1', 'sample2']`
    label = load_dataset(
        "vwxyzjn/lm-human-preferences",
        data_files=[args.label_dataset],
    )["train"]
    print("Num labels found in source:", len(label))
    print("training on", args.labels.num_train, "in batches of", args.local_batch_size)

    if args.normalize_before:
        n_batches = ceil_div(args.normalize_samples, args.local_rollout_batch_size)
        sample_queries_responses = []
        for i in range(n_batches):
            data = next(iter_dataloader)
            queries = data["input_ids"].to(device)
            queries = left_padding_to_right_padding(data["input_ids"], tokenizer.pad_token_id).to(device)
            attention_mask = queries != tokenizer.pad_token_id
            position_ids = attention_mask.cumsum(1) - attention_mask.long() # exclusive cumsum
            output = pretrained_model.generate(
                input_ids=queries,
                attention_mask=attention_mask,
                # position_ids=position_ids, # generation collapsed if this was turned on. TODO: why does generation collapse with this?
                generation_config=generation_config,
                pad_token_id=-1, # disable `pad_token_id` and `eos_token_id` because we just want to
                eos_token_id=-1, # generate tokens without truncation / padding
                return_dict_in_generate=True,
            )
            query_responses = output.sequences
            sample_queries_responses.append(query_responses)

        rewards = []
        for query_responses in sample_queries_responses:
            context_length = queries.shape[1]
            responses = query_responses[:,context_length:]
            attention_mask = query_responses != tokenizer.pad_token_id
            position_ids = attention_mask.cumsum(1) - attention_mask.long() # exclusive cumsum
            output = reward_model.pretrained_model(
                input_ids=query_responses,
                attention_mask=attention_mask,
                position_ids=position_ids,
                return_dict=True,
                output_hidden_states=True,
            )
            reward = reward_model.scalar_head(output.hidden_states[-1])
            reward = reward_model.reward_gain * reward + reward_model.reward_bias
            reward = reward[:, -1]
            rewards.append(reward)
        rewards = torch.cat(rewards)
        mean, std = rewards.mean(), rewards.std()

        # reward normalization
        target_mean, target_std = torch.tensor(0.0, device=device), torch.tensor(1.0, device=device)
        old_gain, old_bias = reward_model.reward_gain, reward_model.reward_bias
        gain = target_std / std
        bias = target_mean - gain * mean
        reward_model.reward_gain.data = gain
        reward_model.reward_bias.data = bias

    print("===training reward model===")
    all_inds = np.arange(args.labels.num_train)
    np.random.shuffle(all_inds)
    global_step = 0
    for start in range(0, args.labels.num_train, args.local_batch_size):
        global_step += 1
        end = start + args.local_batch_size
        b_inds = all_inds[start:end]
        # our_indices = b_inds[rank::self.num_ranks] # TODO: only needed for multi-GPU
        lr = (1 - start / args.labels.num_train) * args.lr
        mb_data = label[b_inds]
        mb_query = torch.from_numpy(np.stack(mb_data["query"])).pin_memory().to(device, non_blocking=True)
        mb_best = torch.from_numpy(np.stack(mb_data["best"])).pin_memory().to(device, non_blocking=True)
        mb_responses = [
            torch.from_numpy(np.stack(mb_data[f"sample{i}"])).pin_memory().to(device, non_blocking=True)
            for i in range(args.labels.num_labels)
        ]
        predicted_rewards = []
        for i in range(args.labels.num_labels):
            mb_query_response = torch.cat([mb_query, mb_responses[i]], dim=1)
            # hack: deal with openai's padding token
            mb_attention_mask = mb_query_response != OPENAI_PAD_TOKEN_ID
            mb_query_response[mb_query_response==OPENAI_PAD_TOKEN_ID] = tokenizer.pad_token_id
            reward = reward_model(
                mb_query_response,
                attention_mask=mb_attention_mask,
            )  # reward has shape (batch_size, sequence_length, 1)
            predicted_rewards.append(
                reward[:, -1].squeeze()
            )  # but we only care about the reward of the last token, resulting in shape (batch_size, 1)

        predicted_rewards = torch.stack(
            predicted_rewards, dim=1
        )  # shape (batch_size, num_labels), basically a reward prediction for each label
        loss = torch.nn.functional.cross_entropy(predicted_rewards, mb_best)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar("loss", loss.item(), global_step)

        if args.print_sample_output_freq > 0 and global_step % args.print_sample_output_freq == 0:
            queries = next(iter_dataloader)
            input_ids = queries["input_ids"][0:1].to(device)
            query_response = pretrained_model.generate(
                input_ids=input_ids,
                attention_mask=queries["attention_mask"][0:1].to(device),
                max_new_tokens=args.task.response_length,
                temperature=args.task.temperature,
                pad_token_id=tokenizer.pad_token_id, # ? should I do this?
            )
            reward = reward_model(
                input_ids=query_response,
                attention_mask=query_response != tokenizer.pad_token_id,
            )[:, -1]
            print(f"global_step {global_step}:")
            print("sample input:", tokenizer.decode(query_response[0][:len(input_ids[0])]).replace("<|endoftext|>", ""))
            print("sample output:", tokenizer.decode(query_response[0][len(input_ids[0]):]).replace("<|endoftext|>", ""))
            print("sample reward:", reward[0].item())

    if args.normalize_after:
        n_batches = ceil_div(args.normalize_samples, args.local_rollout_batch_size)
        sample_queries_responses = []
        for i in range(n_batches):
            data = next(iter_dataloader)
            queries = data["input_ids"].to(device)
            queries = left_padding_to_right_padding(data["input_ids"], tokenizer.pad_token_id).to(device)
            attention_mask = queries != tokenizer.pad_token_id
            position_ids = attention_mask.cumsum(1) - attention_mask.long() # exclusive cumsum
            output = pretrained_model.generate(
                input_ids=queries,
                attention_mask=attention_mask,
                # position_ids=position_ids, # generation collapsed if this was turned on. TODO: why does generation collapse with this?
                generation_config=generation_config,
                pad_token_id=-1, # disable `pad_token_id` and `eos_token_id` because we just want to
                eos_token_id=-1, # generate tokens without truncation / padding
                return_dict_in_generate=True,
            )
            query_responses = output.sequences
            sample_queries_responses.append(query_responses)

        rewards = []
        for query_responses in sample_queries_responses:
            context_length = queries.shape[1]
            responses = query_responses[:,context_length:]
            attention_mask = query_responses != tokenizer.pad_token_id
            position_ids = attention_mask.cumsum(1) - attention_mask.long() # exclusive cumsum
            output = reward_model.pretrained_model(
                input_ids=query_responses,
                attention_mask=attention_mask,
                position_ids=position_ids,
                return_dict=True,
                output_hidden_states=True,
            )
            reward = reward_model.scalar_head(output.hidden_states[-1])
            reward = reward_model.reward_gain * reward + reward_model.reward_bias
            reward = reward[:, -1]
            rewards.append(reward)
        rewards = torch.cat(rewards)
        mean, std = rewards.mean(), rewards.std()

        # reward normalization
        target_mean, target_std = torch.tensor(0.0, device=device), torch.tensor(1.0, device=device)
        old_gain, old_bias = reward_model.reward_gain, reward_model.reward_bias
        gain = target_std / std
        bias = target_mean - gain * mean
        reward_model.reward_gain.data = gain
        reward_model.reward_bias.data = bias

    # save model
    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        torch.save(reward_model.state_dict(), args.save_path)
