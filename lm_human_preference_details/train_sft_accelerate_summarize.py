import os
import random
import time
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import tyro
from accelerate import Accelerator
from datasets import load_dataset
from rich.console import Console
from rich.pretty import pprint
from rich.table import Table
from torch import Tensor, optim
from torch.optim.optimizer import (
    _dispatch_sqrt,
    _get_value,
    _use_grad_for_differentiable,
)
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from lm_human_preference_details.data import process_query


@dataclass
class SFTHParams:
    gradient_accumulation_steps: int = 2
    local_micro_batch_size: int = 8
    noptepochs: int = 1
    lr: float = 6.35e-5
    eps: float = 1e-5
    total_episodes: tyro.conf.Suppress[int] = None
    local_batch_size:tyro.conf.Suppress[int] = None
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
    temperature: float = 0.7


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
    print_sample_output_freq: int = 80
    """How often to print sample output"""
    save_path: str = "models/policy.pt"
    """Where to save the model"""
    use_tensorflow_adam: bool = True
    """Whether to use tensorflow-style Adam optimizer instead of PyTorch's"""
    task: TaskHParams = field(default_factory=TaskHParams)
    sft: SFTHParams = field(default_factory=SFTHParams)


def print_rich_table(title: str, df: pd.DataFrame, console: Console) -> Table:
    table = Table(show_lines=True)
    for column in df.columns:
        table.add_column(column)
    for _, row in df.iterrows():
        table.add_row(*row.astype(str).tolist())
    console.rule(f"[bold red]{title}")
    console.print(table)


def _single_tensor_adam(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    max_exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    amsgrad: bool,
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
    eps: float,
    maximize: bool,
    capturable: bool,
    differentiable: bool,
):
    assert grad_scale is None and found_inf is None

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]
        # update step
        step_t += 1
        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
        step = _get_value(step_t)

        ### pytorch adam implementation:
        # bias_correction1 = 1 - beta1 ** step
        # bias_correction2 = 1 - beta2 ** step
        # step_size = lr / bias_correction1
        # bias_correction2_sqrt = _dispatch_sqrt(bias_correction2)
        # denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
        # param.addcdiv_(exp_avg, denom, value=-step_size)

        ### tensorflow adam implementation:
        lr_t = lr * _dispatch_sqrt(1 - beta2**step) / (1 - beta1**step)
        denom = exp_avg_sq.sqrt().add_(eps)
        param.addcdiv_(exp_avg, denom, value=-lr_t)


def adam(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    max_exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
    foreach: Optional[bool] = None,
    capturable: bool = False,
    differentiable: bool = False,
    fused: Optional[bool] = None,
    grad_scale: Optional[Tensor] = None,
    found_inf: Optional[Tensor] = None,
    *,
    amsgrad: bool,
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
    eps: float,
    maximize: bool,
):
    func = _single_tensor_adam

    func(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        amsgrad=amsgrad,
        beta1=beta1,
        beta2=beta2,
        lr=lr,
        weight_decay=weight_decay,
        eps=eps,
        maximize=maximize,
        capturable=capturable,
        differentiable=differentiable,
        grad_scale=grad_scale,
        found_inf=found_inf,
    )


class AdamTensorFlowStyle(optim.Adam):
    @_use_grad_for_differentiable
    def step(self, closure=None):
        self._cuda_graph_capture_health_check()
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group["betas"]

            self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
            )

            adam(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=group["amsgrad"],
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                maximize=group["maximize"],
                foreach=group["foreach"],
                capturable=group["capturable"],
                differentiable=group["differentiable"],
                fused=group["fused"],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
            )

        return loss


def right_padding_to_left_padding(tokens, pad_id):
    """Convert from right padding to left padding."""
    assert tokens.ndim == 2
    return torch.tensor(
        [[pad_id] * (row == pad_id).sum() + [x for x in row if x != pad_id] for row in tokens],
        device=tokens.device,
    )


def ceil_div(a, b):
    return (a - 1) // b + 1


def exact_div(a, b):
    q = a // b
    if a != q * b:
        raise ValueError(f"Inexact division: {a} / {b} = {a / b}")
    return q


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
        labels=input_ids,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        return_dict=True,
    )


def train(args: Args):
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
    if args.use_tensorflow_adam:
        optimizer = AdamTensorFlowStyle(policy.parameters(), lr=args.sft.lr, eps=args.sft.eps)
    else:
        optimizer = optim.Adam(policy.parameters(), lr=args.sft.lr, eps=args.sft.eps)

    def process_query_data1(x):
        return {
            **process_query(x, encoder=tokenizer, hparams=patch_h),
            "reference_response": tokenizer.encode(
                x["summary"], padding="max_length", max_length=args.task.response_length, truncation=True
            ),
        }

    dataset = dataset.map(process_query_data1)
    dataset = dataset.with_format("torch", columns=["query_token", "reference_response"])
    dataset = dataset.shuffle(seed=local_seed)
    test_dataset = test_dataset.map(process_query_data1)
    test_dataset = test_dataset.with_format("torch", columns=["query_token", "reference_response"])
    test_dataset = test_dataset.shuffle(seed=local_seed)
    dataloader = DataLoader(dataset, batch_size=args.sft.local_batch_size)
    policy, optimizer, dataloader = accelerator.prepare(policy, optimizer, dataloader)
    iter_dataloader = iter(dataloader)
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
    loss_stats = torch.zeros(args.sft.gradient_accumulation_steps, device=device)
    test_data = test_dataset[0:10]
    test_data = {k: v.to(device) for k, v in test_data.items()}
    gradient_accumulation_idx = 0
    for update in range(1, args.sft.num_updates + 1):
        print(update, global_step)
        global_step += 1 * args.sft.batch_size
        frac = 1.0 - (update - 1.0) / args.sft.num_updates
        lrnow = frac * args.sft.lr
        optimizer.param_groups[0]["lr"] = lrnow
        data = next(iter_dataloader)
        queries = data["query_token"].to(device)
        reference_responses = data["reference_response"].to(device)
        query_responses = torch.cat((queries, reference_responses), dim=1)
        query_responses = right_padding_to_left_padding(query_responses, tokenizer.pad_token_id).to(device)
        with accelerator.accumulate(policy):
            output = forward(policy, query_responses, tokenizer)
            loss = output.loss
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
        loss_stats[gradient_accumulation_idx] = loss
        gradient_accumulation_idx = (gradient_accumulation_idx + 1) % args.sft.gradient_accumulation_steps
        if (update - 1) % args.sft.gradient_accumulation_steps:
            writer.add_scalar("loss", accelerator.gather(loss_stats).mean().item(), update)
            writer.add_scalar("lr", lrnow, update)
        if (update - 1) % args.print_sample_output_freq * args.sft.gradient_accumulation_steps == 0:
            with torch.no_grad():
                test_queries = test_data["query_token"]
                test_reference_responses = test_data["reference_response"]
                test_queries = right_padding_to_left_padding(test_queries, tokenizer.pad_token_id)
                generated_responses = generate(policy, test_queries, tokenizer, generation_config)

                try:
                    all_decode_test_queries = tokenizer.batch_decode(test_queries, skip_special_tokens=True)
                    all_decode_test_query_responses = tokenizer.batch_decode(generated_responses, skip_special_tokens=True)
                    all_decode_test_reference_responses = tokenizer.batch_decode(
                        test_reference_responses, skip_special_tokens=True
                    )
                    all_decode_test_responses = [
                        x[len(y) :] for x, y in zip(all_decode_test_query_responses, all_decode_test_queries)
                    ]

                    all_df = pd.DataFrame(
                        {
                            "query": all_decode_test_queries,
                            "response": all_decode_test_responses,
                            "reference": all_decode_test_reference_responses,
                        }
                    )
                    if accelerator.is_main_process and args.track:
                        wandb.log({"query_responses": wandb.Table(dataframe=all_df)}, step=update)
                    print_rich_table(f"Sample Output at Step {update}", all_df[:4], console)
                except Exception as e:
                    print(e)

    # save model
    if accelerator.is_main_process and args.save_path:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        torch.save(policy.state_dict(), args.save_path)

        if args.upload_model:
            repo_name = f"{args.exp_name}__{args.rewards.label_dataset}__seed{args.seed}__{int(time.time())}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            policy.save_pretrained(repo_id, safe_serialization=True, push_to_hub=True)
            tokenizer.save_pretrained(repo_id, push_to_hub=True)

if __name__ == "__main__":
    args = tyro.cli(Args)
    train(args)
