import os
import random
import time
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch
import torch.optim as optim
import tyro
from accelerate import Accelerator
from datasets import load_dataset
from rich.console import Console
from rich.pretty import pprint
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer

from lm_human_preference_details.data import process_query


@dataclass
class SFTHParams:
    gradient_accumulation_steps: int = 1
    local_micro_batch_size: int = 16
    noptepochs: int = 1
    lr: float = 6.35e-5
    eps: float = 1e-5
    lm_loss_on_response_only: bool = False
    total_episodes: tyro.conf.Suppress[int] = None
    local_batch_size: tyro.conf.Suppress[int] = None
    batch_size: tyro.conf.Suppress[int] = None
    mini_batch_size: tyro.conf.Suppress[int] = None
    world_size: tyro.conf.Suppress[int] = None
    batch_size: tyro.conf.Suppress[int] = None
    num_updates: tyro.conf.Suppress[int] = None


@dataclass
class TaskHParams:
    # Query params
    query_length: int = 512
    query_dataset: str = "vwxyzjn/summarize_from_feedback_tldr_3_filtered"

    query_format_str: Optional[str] = "SUBREDDIT: r/{subreddit}\n\nTITLE: {title}\n\nPOST: {post}\n\nTL;DR:"
    query_truncate_field: Optional[str] = "post"
    query_truncate_text: Optional[str] = "\n"
    query_padding: Optional[str] = None  # defaults to repeated spaces
    query_pad_side: Optional[str] = "left"

    # Response params
    response_length: int = 48

    # Truncate response after the first occurrence of this token at or after index after when sampling.
    truncate_token: int = 50256  # EOS token
    truncate_after: int = 16
    penalty_reward_value: int = -1

    # LM params
    temperature: float = 0.01


# a patch
@dataclass
class TaskQueryHParams:
    length: int = None
    dataset: str = None
    format_str: Optional[str] = None  # if underlying dataset yields dicts, can format arbitrarily
    truncate_field: Optional[str] = None
    truncate_text: Optional[str] = None
    padding: Optional[str] = None  # defaults to repeated spaces
    pad_side: Optional[str] = None


@dataclass
class Args:
    # common args
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
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
    upload_model: bool = False
    "whether to upload the saved model to huggingface"
    hf_entity: str = ""
    "the user or org name of the model repository from the Hugging Face Hub"

    base_model: str = "gpt2"
    """the name of the pretrained model to use"""
    deepspeed: bool = False
    """Whether to use deepspeed to train the model"""
    print_sample_output_freq: int = 220
    """How often to print sample output"""
    save_path: str = "models/sft_policy.pt"
    """Where to save the model"""
    use_tensorflow_adam: bool = True
    """Whether to use tensorflow-style Adam optimizer instead of PyTorch's"""
    task: TaskHParams = field(default_factory=TaskHParams)
    sft: SFTHParams = field(default_factory=SFTHParams)


def right_padding_to_left_padding(tokens, pad_id):
    """Convert from right padding to left padding."""
    assert tokens.ndim == 2
    return torch.tensor(
        [[pad_id] * (row == pad_id).sum() + [x for x in row if x != pad_id] for row in tokens],
        device=tokens.device,
    )


def generate(lm_backbone, queries, tokenizer, generation_config):
    """generate in a way that does not affect padding tokens"""
    context_length = queries.shape[1]
    attention_mask = queries != tokenizer.pad_token_id
    input_ids = queries.clone()
    input_ids[~attention_mask] = 0  # set padding tokens to 0
    output = lm_backbone.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        # position_ids=attention_mask.cumsum(1) - attention_mask.long(), # generation collapsed if this was turned on. TODO: why does generation collapse with this?
        generation_config=generation_config,
        return_dict_in_generate=True,
    )
    # restore padding tokens
    return torch.cat((queries, output.sequences[:, context_length:]), dim=1)


def forward(policy, query_responses, tokenizer):
    attention_mask = query_responses != tokenizer.pad_token_id
    position_ids = attention_mask.cumsum(1) - attention_mask.long()  # exclusive cumsum
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    return policy(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        return_dict=True,
    )


if __name__ == "__main__":
    args = tyro.cli(Args)
    accelerator = Accelerator(gradient_accumulation_steps=args.sft.gradient_accumulation_steps)
    args.sft.world_size = accelerator.num_processes
    args.sft.local_batch_size = args.sft.local_micro_batch_size * args.sft.gradient_accumulation_steps
    args.sft.batch_size = int(args.sft.local_batch_size * args.sft.world_size)
    patch_h = TaskQueryHParams(
        length=args.task.query_length,
        dataset=args.task.query_dataset,
        format_str=args.task.query_format_str,
        truncate_field=args.task.query_truncate_field,
        truncate_text=args.task.query_truncate_text,
        padding=args.task.query_padding,
        pad_side=args.task.query_pad_side,
    )
    dataset = load_dataset(args.task.query_dataset, split="train")
    test_dataset = load_dataset(args.task.query_dataset, split="test")
    accelerator.print("The number of samples in dataset", len(dataset))
    accelerator.print("The number of samples in test_dataset", len(test_dataset))
    args.sft.total_episodes = len(dataset)
    args.sft.num_updates = args.sft.total_episodes // args.sft.batch_size

    console = Console(force_terminal=True)
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SimpleNamespace()  # dummy writer
    writer.add_scalar = lambda x, y, z: None
    writer.add_histogram = lambda x, y, z: None
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
    local_seed = args.seed + accelerator.process_index * 100003  # Prime
    random.seed(local_seed)
    np.random.seed(local_seed)
    torch.manual_seed(local_seed)
    torch.backends.cudnn.deterministic = True
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        padding_side="right",
        trust_remote_code=True,
    )
    # we use the padding token manually but do not resize the token embedding of the model
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    policy = AutoModelForCausalLM.from_pretrained(args.base_model, trust_remote_code=True)
    policy.generation_config.eos_token_id = None  # disable `pad_token_id` and `eos_token_id` because we just want to
    policy.generation_config.pad_token_id = None  # generate tokens without truncation / padding
    # IMPORTANT: Layer norm produces weird gradients, which affects Adam optimizer to impact all the parameters systematically
    # see https://github.com/pytorch/pytorch/issues/104857 for more details
    optimizer = optim.Adam(policy.parameters(), lr=args.sft.lr, eps=args.sft.eps)

    def process_query_data(x):
        return {
            **process_query(x, encoder=tokenizer, hparams=patch_h),
            "reference_response": tokenizer.encode(
                f" {x['summary']}<|endoftext|>",
                padding="max_length",
                max_length=args.task.response_length,
                truncation=True,
                # with an extra leading space to account for the space between the query and response
            ),
        }

    dataset = dataset.map(process_query_data)
    dataset = dataset.with_format("torch", columns=["query_token", "reference_response"])
    dataset = dataset.shuffle(seed=local_seed)
    test_dataset = test_dataset.map(process_query_data)
    test_dataset = test_dataset.with_format("torch", columns=["query_token", "reference_response"])
    test_dataset = test_dataset.shuffle(seed=local_seed)
    dataloader = DataLoader(dataset, batch_size=args.sft.local_micro_batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=args.sft.local_micro_batch_size)
    policy, optimizer, dataloader, test_dataloader = accelerator.prepare(policy, optimizer, dataloader, test_dataloader)
    iter_dataloader = iter(dataloader)
    # WARNING: even with `max_new_tokens` and `min_new_tokens` set to the same value, the number of tokens generated
    # may not be the same. TODO: investigate further, we just want to generate a fixed number of tokens
    # generation_config = GenerationConfig(
    #     max_new_tokens=args.task.response_length,
    #     min_new_tokens=args.task.response_length,
    #     temperature=args.task.temperature,
    #     top_k=0.0,
    #     top_p=1.0,
    #     do_sample=True,
    # )

    print("===training policy===")
    global_step = 0
    test_data = test_dataset[0:10]
    test_data = {k: v.to(device) for k, v in test_data.items()}

    # Given parameters
    eta_min = 0
    eta_max = 6.35e-5
    T_max = args.sft.num_updates

    for update in range(1, args.sft.num_updates + 1):
        global_step += 1 * args.sft.batch_size
        accelerator.print(f"update {update}, global_step {global_step}")
        # frac = 1.0 - (update - 1.0) / args.sft.num_updates
        # lrnow = frac * args.sft.lr
        lrnow = eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * (update - 1) / T_max))
        optimizer.param_groups[0]["lr"] = lrnow
        data = next(iter_dataloader)
        queries = data["query_token"].to(device)
        reference_responses = data["reference_response"].to(device)
        queries = right_padding_to_left_padding(queries, tokenizer.pad_token_id).to(device)
        query_responses = torch.cat((queries, reference_responses), dim=1)
        with accelerator.accumulate(policy):
            output = forward(policy, query_responses, tokenizer)
            # mask out gradient effects on response padding tokens
            labels = query_responses.masked_fill(query_responses == tokenizer.pad_token_id, -1)
            if args.sft.lm_loss_on_response_only:
                # mask out gradient effects on query tokens
                labels[:, : queries.shape[1]] = -1
            lm_logits = output.logits
            # hand-rolled transformer loss: Shift so that tokens < n predict n
            # but unlike `transformers` we mask the padding tokens via `ignore_index=-1`
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-1)
            raise
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
