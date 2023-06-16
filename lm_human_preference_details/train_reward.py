from dataclasses import asdict, dataclass, field
import os
import subprocess
import time
from typing import Optional
import numpy as np

import tyro
import torch
import torch.optim as optim
import torch.nn as nn
from datasets import load_dataset


from torch.utils.tensorboard import SummaryWriter
from rich.pretty import pprint
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from torch.utils.data import IterableDataset, DataLoader
from lm_human_preference_details.data import DATASET

@dataclass
class LabelHParams:
    type: str = None
    num_train: int = 4992
    num_labels: int = 4
    source: str = None


@dataclass
class LMHParams:
    # Query params
    query_length: int = 64
    query_dataset: str = "books"
    query_prefix: str = ''
    query_suffix: str = ''
    start_text: Optional[str] = None
    end_text: Optional[str] = None

    # Response params
    response_length: int = 24

    # Truncate response after the first occurrence of this token at or after index after when sampling.
    truncate_token: Optional[int] = None
    truncate_after: int = 0
    penalty_reward_value: int = -1

    # LM params
    temperature: float = 1.0
    initial_model: str = None


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

    task: LMHParams = field(default_factory=LMHParams)
    labels: LabelHParams = field(default_factory=LabelHParams)
    base_model: str = "gpt2"

    local_batch_size: int = 32
    lr: float = 0.00005

    local_rollout_batch_size: int = 64 # per rank (8 total ranks)
    normalize_samples: int = 256  # Samples used to estimate reward mean and std
    debug_normalize: int = 0  # Samples used to check that normalization worked

    normalize_before: bool = True
    """Whether, before training, to normalize the rewards on the policy to the scales on the training buffer. (For comparisons, just use mean 0, var 1.)"""
    normalize_after: bool = True
    """Whether, after training, to normalize the rewards on the ref policy to mean 0, var 1 (so the KL coefficient always has the same meaning)."""
    cuda: bool = True
    """Whether to use cuda if available."""



OPENAI_PAD_TOKEN_ID= 50259
OUR_PAD_TOKEN_ID = 50257
class ValueHead(nn.Module):
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

        self.summary = nn.Linear(hidden_size, 1)

        self.flatten = nn.Flatten()

    def forward(self, hidden_states):
        output = self.dropout(hidden_states)

        # For now force upcast in fp32 if needed. Let's keep the
        # output in fp32 for numerical stability.
        if output.dtype != self.summary.weight.dtype:
            output = output.to(self.summary.weight.dtype)

        output = self.summary(output)
        return output


class AutoModelForCausalLMWithValueHead(nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.value_head = ValueHead(self.pretrained_model.config)

    def forward(self, input_ids):
        output = self.pretrained_model(input_ids, output_hidden_states=True)
        value = self.value_head(output.hidden_states[-1])
        return value


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
        self.pad_token = token_to_index[tokenizer.pad_token]


    def __iter__(self):
        for text in self.generator("train", args.seed):
            tokens = self.tokenizer.encode(text)
            if self.start_token is not None:
                try:
                    first_index = tokens.index(self.start_token)+1
                    if first_index < len(tokens):
                        tokens = tokens[first_index:]
                except:
                    continue
            tokens = tokens[:self.query_length]
            if self.end_token is not None:
                try:
                    last_index = len(tokens)-tokens[::-1].index(self.end_token)
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


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.track:
        # git_tag = subprocess.check_output(["git", "describe", "--tags"]).decode("ascii").strip()
        # os.environ["WANDB_TAGS"] = git_tag
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
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, 
        padding_side="right", # TODO: What is this? It avoids `A decoder-only architecture is being used, but right-padding was detected! For correct generation results, please set `padding_side='left'` when initializing the tokenizer.`
    )
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    pretrained_model = AutoModelForCausalLM.from_pretrained(args.base_model).to(device)
    pretrained_model.resize_token_embeddings(len(tokenizer))
    # tokenizer.pad_token = tokenizer.eos_token
    reward_model = AutoModelForCausalLMWithValueHead(pretrained_model).to(device)
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

    label = load_dataset(
        "vwxyzjn/lm-human-preferences",
        data_files=["sentiment/offline_5k.json"]
    )["train"]
    # `label` has keys `['sample0', 'query', 'best', 'sample3', 'sample1', 'sample2']`
    def replace_pad_with_eos(data):
        # openai lm-human-preferences uses 50259 for <pad>
        # https://github.com/openai/lm-human-preferences/blob/cbfd210bb8b08f6bc5c26878c10984b90f516c66/lm_human_preferences/language/encodings.py#L56
        for key in ['sample0', 'query', 'sample3', 'sample1', 'sample2']:
            for i in range(len(data[key])):
                if data[key][i] == OPENAI_PAD_TOKEN_ID:
                    data[key][i] = tokenizer.pad_token_id
        return data
    label = label.map(replace_pad_with_eos)
    assert np.array(label["query"]).max() != OPENAI_PAD_TOKEN_ID
    print('Num labels found in source:', len(label))
    print("training on", args.labels.num_train, "in batches of", args.local_batch_size)

    if args.normalize_before:
        pass

    all_inds = np.arange(args.labels.num_train)
    np.random.shuffle(all_inds)
    global_step = 0
    for epoch in range(2):
        for start in range(0, args.labels.num_train, args.local_batch_size):
            global_step += 1
            end = start + args.local_batch_size
            b_inds = all_inds[start:end]
            # our_indices = b_inds[rank::self.num_ranks] # TODO: only needed for multi-GPU
            lr = (1 - start / args.labels.num_train) * args.lr
            mb_data = label[b_inds]
            mb_query = torch.from_numpy(np.stack(mb_data["query"])).pin_memory().to(device, non_blocking=True)
            mb_best = torch.from_numpy(np.stack(mb_data["best"])).pin_memory().to(device, non_blocking=True)
            values = []
            for i in range(args.labels.num_labels):
                with torch.no_grad():
                    mb_response = reward_model.pretrained_model.generate(
                        input_ids=mb_query,
                        max_new_tokens=args.task.response_length,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                value = reward_model(torch.cat([mb_query, mb_response], dim=1)) # value has shape (batch_size, sequence_length, 1)
                values.append(value[:, -1].squeeze()) # but we only care about the value of the last token
            
            values = torch.stack(values, dim=1) # shape (batch_size, num_labels), basically a value prediction for each label
            loss = torch.nn.functional.cross_entropy(values, mb_best)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            writer.add_scalar("loss", loss.item(), global_step)

    # raise
    if args.normalize_after:
        pass

    # STRING_MULTIPLIER = 20 # otherwise our toy string is too short
    # reward_model(tokenizer.encode("you fucking sucks " * STRING_MULTIPLIER, return_tensors="pt").to(device))[:, -1]
    # reward_model(tokenizer.encode("you are great " * STRING_MULTIPLIER, return_tensors="pt").to(device))[:, -1]
    # reward_model(tokenizer.encode("you are stupid " * STRING_MULTIPLIER, return_tensors="pt").to(device))[:, -1]
    # reward_model(tokenizer.encode("you are beautiful " * STRING_MULTIPLIER, return_tensors="pt").to(device))[:, -1]

# def debug(i):
#     print("query", tokenizer.decode(np.stack(mb_data["query"])[i]))
#     print("sample0", tokenizer.decode(np.stack(mb_data["sample0"])[i]))
#     print("sample1", tokenizer.decode(np.stack(mb_data["sample1"])[i]))
#     print("sample2", tokenizer.decode(np.stack(mb_data["sample2"])[i]))
#     print("sample3", tokenizer.decode(np.stack(mb_data["sample3"])[i]))
