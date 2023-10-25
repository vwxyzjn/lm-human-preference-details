import os
import random
import time
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace
from typing import List, Literal, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import transformers
import tyro
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils import DistributedDataParallelKwargs, broadcast
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
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, get_scheduler

from lm_human_preference_details.data import process_query


@dataclass
class LabelHParams:
    type: str = None
    num_train: int = 92832
    num_labels: int = 2
    source: str = None


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
    load_from_cache_file: bool = False
    """Whether to load data from the local cache file in `dataset.map`"""

    base_model: str = "gpt2"
    """the name of the pretrained model to use"""
    deepspeed: bool = False
    """Whether to use deepspeed to train the model"""
    label_dataset: str = "openai/summarize_from_feedback"
    """the name of the dataset to use for labels in `https://huggingface.co/datasets/vwxyzjn/lm-human-preferences`"""
    local_batch_size: int = 4
    """per rank batch size"""
    gradient_accumulation_steps: int = 1
    """gradient accumulation steps"""
    local_micro_batch_size: tyro.conf.Suppress[int] = None
    """per rank micro batch size"""
    lr: float = 0.00005
    """the learning rate"""
    eps: float = 1e-5
    """the epsilon for AdamW"""
    local_rollout_batch_size: int = 512
    """per rank rollout batch size"""
    rollout_batch_size: tyro.conf.Suppress[int] = None
    """rollout batch size"""
    world_size: tyro.conf.Suppress[int] = None
    """the number of processes to use"""
    batch_size: tyro.conf.Suppress[int] = None
    """the batch size across all ranks"""
    local_normalize_samples: int = 256
    """Samples used to estimate reward mean and std"""
    normalize_samples: tyro.conf.Suppress[int] = None
    """Samples used to estimate reward mean and std across all ranks"""
    debug_normalize: int = 0
    """Samples used to check that normalization worked"""
    normalize_before: bool = True
    """Whether, before training, to normalize the rewards on the policy to the scales on the training buffer. (For comparisons, just use mean 0, var 1.)"""
    normalize_after: bool = True
    """Whether, after training, to normalize the rewards on the ref policy to mean 0, var 1 (so the KL coefficient always has the same meaning)."""
    print_sample_output_freq: int = 300
    """How often to print sample output"""
    sft_model_path: str = "models/sft_policy"
    """Where to load the SFT model"""
    logsigmoid: bool = True
    """Whether to use log-sigmoid loss instead of cross-entropy loss"""
    trainable_param_percentage: float = 1.0
    """Percentage of parameters to train"""
    num_epochs: int = 1
    """Number of epochs to train"""
    num_updates: tyro.conf.Suppress[int] = None
    """Number of updates to train"""
    save_path: str = "models/reward"
    """Where to save the model"""
    optimizer: Literal["tf_adam", "adam", "adamw"] = "adamw"
    """Which optimizer to use"""
    scheduler: str = "constant_with_warmup"
    """Which scheduler to use"""
    warm_up_steps: int = 100
    """Number of warm up steps for the scheduler"""
    task: TaskHParams = field(default_factory=TaskHParams)
    labels: LabelHParams = field(default_factory=LabelHParams)


def first_true_indices(bools, dtype=torch.long):
    """
    Takes an N-dimensional bool tensor and returns an (N-1)-dimensional tensor of integers giving
    the position of the first True in each "row".

    Returns the length of the rows (bools.size(-1)) if no element is True in a given row.
    """
    row_len = bools.size(-1)
    zero_or_index = row_len * (~bools).type(dtype) + torch.arange(row_len, dtype=dtype, device=bools.device)
    return torch.min(zero_or_index, dim=-1).values


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


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.normal_(layer.weight, std=std)
    torch.nn.init.constant_(layer.bias, val=bias_const)
    return layer


class AutoModelForCausalLMWithRewardHead(nn.Module):
    def __init__(self, lm_backbone):
        super().__init__()
        self.lm_backbone = lm_backbone
        self.scalar_head = layer_init(
            nn.Linear(lm_backbone.config.hidden_size, 1),
            std=1 / np.sqrt(lm_backbone.config.hidden_size + 1),
        )
        # self.reward_gain = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
        # self.reward_bias = torch.nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def forward(self, **kwargs):
        output = self.lm_backbone(**kwargs)
        last_reward_latents = output.hidden_states[-1]
        # shape: [batch_size, hidden_size]
        reward = self.scalar_head(last_reward_latents)
        return output, reward


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


def get_reward(reward_model, query_responses, tokenizer):
    attention_mask = query_responses != tokenizer.pad_token_id
    position_ids = attention_mask.cumsum(1) - attention_mask.long()  # exclusive cumsum
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    return reward_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        return_dict=True,
        output_hidden_states=True,
    )


def get_reward_complete(reward_model, query_responses, tokenizer):
    reward = get_reward(reward_model, query_responses, tokenizer)[1]
    last_response_indices = first_true_indices(query_responses == tokenizer.pad_token_id) - 1
    last_response_indices = torch.max(
        last_response_indices,
        torch.zeros([1], dtype=last_response_indices.dtype, device=query_responses.device),
    )
    return reward[:, :, 0].gather(1, last_response_indices.unsqueeze(1)).view(-1), reward


def normalize(
    tokenizer,
    accelerator,
    device,
    lm_backbone,
    reward_model,
    dataloader,
    validation_dataloader,
):
    idx = 0
    with torch.no_grad():
        # reset reward scales
        accelerator.unwrap_model(reward_model).reward_gain.data.fill_(1.0)
        accelerator.unwrap_model(reward_model).reward_bias.data.fill_(0.0)
        # number of minibatches for computing the normalization statistics
        rewards = []
        for data in dataloader:
            idx += len(data["query_token"])
            queries = data["query_token"].to(device)
            queries = right_padding_to_left_padding(queries, tokenizer.pad_token_id).to(device)
            reference_response = data["reference_response"].to(device)
            query_responses = torch.cat((queries, reference_response), dim=1)
            score = get_reward_complete(reward_model, query_responses, tokenizer)
            rewards.append(score)
        accelerator.print(f"====number of samples per device: {idx}")
        rewards = torch.cat(rewards)
        rewards = accelerator.gather(rewards)
        mean, std = rewards.mean(), rewards.std()
        print(f"mean: {mean}, std: {std}")

        # reward normalization
        target_mean, target_std = torch.tensor(0.0, device=device), torch.tensor(1.0, device=device)
        gain = target_std / std
        bias = target_mean - gain * mean
        print(f"gain: {gain}, bias: {bias}")
        accelerator.unwrap_model(reward_model).reward_gain.data = gain
        accelerator.unwrap_model(reward_model).reward_bias.data = bias

        # validate normalization
        rewards = []
        for data in validation_dataloader:
            queries = data["query_token"].to(device)
            queries = right_padding_to_left_padding(queries, tokenizer.pad_token_id).to(device)
            reference_response = data["reference_response"].to(device)
            query_responses = torch.cat((queries, reference_response), dim=1)
            score = get_reward_complete(reward_model, query_responses, tokenizer)
            rewards.append(score)
        rewards = torch.cat(rewards)
        rewards = accelerator.gather(rewards)
        mean, std = rewards.mean(), rewards.std()
        print(f"after mean: {mean}, after std: {std}")


def evaluate(args, accelerator, device, reward_model, validation_label):
    # reward_model.eval()
    with torch.no_grad():
        # eval on validation_label, some duplicate code (I don't want to make the training loop into a function...)
        test_accuracies = []
        eval_len = len(validation_label)
        len_labels = (eval_len // args.batch_size) * args.batch_size  # in case the last batch is not full
        new_all_inds = np.arange(len_labels)
        for start in range(0, len_labels, args.batch_size):
            end = start + args.batch_size
            b_inds_all = new_all_inds[start:end]
            b_inds = b_inds_all[accelerator.process_index :: accelerator.num_processes]  #  multi-GPU slicing
            for micro_batch_start in range(0, args.local_batch_size, args.local_micro_batch_size):
                micro_batch_end = micro_batch_start + args.local_micro_batch_size
                micro_batch_inds = b_inds[micro_batch_start:micro_batch_end]
                mb_data = validation_label[micro_batch_inds]
                mb_query = torch.from_numpy(np.stack(mb_data["query_token"])).to(device)
                mb_query = right_padding_to_left_padding(mb_query, args.pad_token_id).to(device)
                mb_best = torch.from_numpy(np.stack(mb_data["choice"])).to(device)
                mb_responses = [
                            torch.from_numpy(np.stack(mb_data[f"response{i}_token"])).to(device)
                            for i in range(args.labels.num_labels)
                        ]
                predicted_reward = []
                for i in range(args.labels.num_labels):
                    query_responses = torch.cat([mb_query, mb_responses[i]], dim=1)
                    score, _ = get_reward_complete(reward_model, query_responses, args)
                    predicted_reward.append(score)
                predicted_reward = torch.stack(
                    predicted_reward, dim=1
                )  # shape (batch_size, num_labels), basically a reward prediction for each label
                accuracy = (predicted_reward.argmax(1) == mb_best).float().mean()
                test_accuracies.append(accuracy)
        test_accuracy = accelerator.gather(torch.stack(test_accuracies).mean()).mean().item()
    # reward_model.train()
    return test_accuracy


def train(args: Args):
    accelerator = Accelerator(
        kwargs_handlers=[
            DistributedDataParallelKwargs(
                broadcast_buffers=False,
                # find_unused_parameters=True,
            )
        ],  # this is needed to avoid https://github.com/pytorch/pytorch/issues/22095#issuecomment-505099500
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    args.world_size = accelerator.num_processes
    args.batch_size = int(args.local_batch_size * args.world_size)
    args.rollout_batch_size = int(args.local_rollout_batch_size * args.world_size)
    args.local_micro_batch_size = exact_div(args.local_batch_size, args.gradient_accumulation_steps)
    args.num_updates = args.labels.num_train // args.batch_size
    patch_h = TaskQueryHParams(
        length=args.task.query_length,
        dataset=args.task.query_dataset,
        format_str=args.task.query_format_str,
        truncate_field=args.task.query_truncate_field,
        truncate_text=args.task.query_truncate_text,
        padding=args.task.query_padding,
        pad_side=args.task.query_pad_side,
    )

    console = Console(force_terminal=True)
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SimpleNamespace()  # dummy writer
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
    args.pad_token_id = tokenizer.pad_token_id
    reward_model = AutoModelForCausalLMWithRewardHead(
        AutoModelForCausalLM.from_pretrained(args.base_model, trust_remote_code=True)
    )

    # freeze the first 70% of layers
    if args.trainable_param_percentage < 1.0:
        layers = reward_model.lm_backbone.transformer.h
        num_layers = len(layers)
        num_unfrozen = int(args.trainable_param_percentage * num_layers)
        for layer in layers[:-num_unfrozen]:
            layer.requires_grad_(False)

    if args.sft_model_path:
        reward_model.lm_backbone.load_state_dict(torch.load(args.sft_model_path, map_location=device))
        print(f"loaded SFT model from {args.sft_model_path}")
    reward_model.lm_backbone.generation_config.eos_token_id = (
        None  # disable `pad_token_id` and `eos_token_id` because we just want to
    )
    reward_model.lm_backbone.generation_config.pad_token_id = None  # generate tokens without truncation / padding
    # make sure the `lm_head` or `embed_out` does not require gradients, otherwise
    # pytorch DDP complains; see https://gist.github.com/vwxyzjn/45fc8706dfb3cf33695f0f57cc44a533
    if isinstance(reward_model.lm_backbone, transformers.GPTNeoXForCausalLM):
        reward_model.lm_backbone.embed_out.requires_grad_(False)
    if args.optimizer == "tf_adam":
        optimizer = AdamTensorFlowStyle(reward_model.parameters(), lr=args.lr, eps=args.eps)
    elif args.optimizer == "adam":
        optimizer = optim.Adam(reward_model.parameters(), lr=args.lr, eps=args.eps)
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW(reward_model.parameters(), lr=args.lr, eps=args.eps)
    # TODO: use AdamW
    scheduler = get_scheduler(
        args.scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.warm_up_steps,
        num_training_steps=args.num_updates * args.num_epochs,
    )

    if args.deepspeed:
        import deepspeed

        deepspeed_states = AcceleratorState().deepspeed_plugin
        deepspeed_states.deepspeed_config["train_micro_batch_size_per_gpu"] = args.local_micro_batch_size


    reward_model, optimizer, scheduler = accelerator.prepare(reward_model, optimizer, scheduler)
    if args.normalize_before:
        dataset = load_dataset(args.task.query_dataset, split="train")
        validation_dataset = load_dataset(args.task.query_dataset, split="validation")

        def process_query_data(x):
            return {
                **process_query(x, encoder=tokenizer, hparams=patch_h),
                "reference_response": tokenizer.encode(
                    f" {x['summary']}<|endoftext|>", padding="max_length", max_length=args.task.response_length, truncation=True,
                    # with an extra leading space to account for the space between the query and response
                ),
            }

        dataset = dataset.map(process_query_data, load_from_cache_file=args.load_from_cache_file)
        dataset = dataset.with_format("torch", columns=["query_token", "reference_response"])
        dataset = dataset.shuffle(seed=local_seed)
        dataloader = DataLoader(dataset, batch_size=args.local_rollout_batch_size)
        validation_dataset = validation_dataset.map(process_query_data, load_from_cache_file=args.load_from_cache_file)
        validation_dataset = validation_dataset.with_format("torch", columns=["query_token", "reference_response"])
        validation_dataset = validation_dataset.shuffle(seed=local_seed)
        validation_dataloader = DataLoader(validation_dataset, batch_size=args.local_rollout_batch_size)
        dataloader = accelerator.prepare(dataloader)
        iter_dataloader = iter(dataloader)
        print("===Normalize reward model *before* training===")
        print(
            "before normalization. "
            + f"Gain: {accelerator.unwrap_model(reward_model).reward_gain.data}"
            + f" Bias: {accelerator.unwrap_model(reward_model).reward_bias.data}"
        )

        normalize(
            tokenizer,
            accelerator,
            device,
            reward_model,
            reward_model,
            dataloader,
            validation_dataloader,
        )
        print(
            "after normalization. "
            + f"Gain: {accelerator.unwrap_model(reward_model).reward_gain.data}"
            + f" Bias: {accelerator.unwrap_model(reward_model).reward_bias.data}"
        )

    # `label` has keys `['sample0', 'query', 'best', 'sample3', 'sample1', 'sample2']`
    label = load_dataset(args.label_dataset, "comparisons", split="train")
    validation_label = load_dataset(args.label_dataset, "comparisons", split="validation")
    dev_validation_label = validation_label.filter(lambda x: x["split"] == "valid1")
    eval_validation_label = validation_label.filter(lambda x: x["split"] == "valid2")
    accelerator.print("Num labels found in source:", len(label))
    accelerator.print("training on", args.labels.num_train, "in batches of", args.local_batch_size)

    def process_response_data(x):
        return {
            **process_query(x["info"], encoder=tokenizer, hparams=patch_h),
            "response0_token": tokenizer.encode(
                f" {x['summaries'][0]['text']}<|endoftext|>", padding="max_length", max_length=args.task.response_length, truncation=True
            ),
            "response1_token": tokenizer.encode(
                f" {x['summaries'][1]['text']}<|endoftext|>", padding="max_length", max_length=args.task.response_length, truncation=True
            ),
        }

    label = label.map(process_response_data, load_from_cache_file=args.load_from_cache_file)
    dev_validation_label = dev_validation_label.map(process_response_data, load_from_cache_file=args.load_from_cache_file)
    eval_validation_label = eval_validation_label.map(process_response_data, load_from_cache_file=args.load_from_cache_file)
    # TODO: check if all labels have eos token
    accelerator.print("===training reward model===")
    num_train = (args.labels.num_train // args.batch_size) * args.batch_size
    for epoch in range(args.num_epochs):
        all_inds = np.random.permutation(args.labels.num_train)
        # ensure that all processes have the same shuffled indices
        all_inds = broadcast(torch.tensor(all_inds, device=device), 0)
        all_inds = all_inds.cpu().numpy()
        accelerator.print(f"epoch: {epoch}")
        for (epoch_global_step, start) in enumerate(range(0, num_train, args.batch_size)):
            global_step = epoch * args.num_updates + epoch_global_step
            end = start + args.batch_size
            b_inds_all = all_inds[start:end]
            b_inds = b_inds_all[accelerator.process_index :: accelerator.num_processes]  #  multi-GPU slicing
            # accelerator.print(f"global_step: {global_step}, start: {start}, end: {end}, b_inds: {b_inds}")
            if accelerator.is_main_process: pprint(
                {
                    "global_step": global_step,
                    "start:end": f"{start}:{end}",
                    "b_inds_all": b_inds_all,
                    "b_inds": b_inds,
                }
            )
            losses = torch.zeros((args.gradient_accumulation_steps,), device=device)
            accuracies = torch.zeros((args.gradient_accumulation_steps,), device=device)
            reward_preferreds = torch.zeros((args.gradient_accumulation_steps,), device=device)
            reward_rejecteds = torch.zeros((args.gradient_accumulation_steps,), device=device)
            gradient_accumulation_step = 0
            # reward_model.train()
            for micro_batch_start in range(0, args.local_batch_size, args.local_micro_batch_size):
                with accelerator.accumulate(reward_model):
                    micro_batch_end = micro_batch_start + args.local_micro_batch_size
                    micro_batch_inds = b_inds[micro_batch_start:micro_batch_end]
                    mb_data = label[micro_batch_inds]
                    # pprint({
                    #     "micro_batch_start:micro_batch_end": f"{micro_batch_start}:{micro_batch_end}",
                    #     "micro_batch_inds": micro_batch_inds,
                    # })
                    mb_query = torch.from_numpy(np.stack(mb_data["query_token"])).to(device)
                    mb_best = torch.from_numpy(np.stack(mb_data["choice"])).to(device)
                    mb_responses = [
                        torch.from_numpy(np.stack(mb_data[f"response{i}_token"])).to(device) for i in range(args.labels.num_labels)
                    ]
                    mb_query_tiled = mb_query.unsqueeze(1).repeat(1, len(mb_responses), 1)
                    query_responses = torch.cat([mb_query_tiled, torch.stack(mb_responses).transpose(0,1)], dim=2).flatten(0, 1)
                    predicted_reward, reward = get_reward_complete(reward_model, query_responses, tokenizer)
                    predicted_reward = predicted_reward.view(-1, len(mb_responses)) # TODO check shape for no gradienta ccumulation steps
                    
                    # print(tokenizer.decode(mb_query[0]))
                    # print(tokenizer.decode(mb_responses[0][0]))
                    # print(tokenizer.decode(mb_responses[1][0]))
                    # predicted_reward = []
                    # rewards = []
                    # for i in range(args.labels.num_labels):
                    #     query_responses = torch.cat([mb_query, mb_responses[i]], dim=1)
                    #     score, reward = get_reward_complete(reward_model, query_responses, tokenizer)
                    #     rewards.append(reward.squeeze(-1))
                    #     predicted_reward.append(score)
                    # # shape (batch_size, num_labels), basically a reward prediction for each label
                    # predicted_reward = torch.stack(predicted_reward, dim=1)
                    # breakpoint()
                    accuracy = (predicted_reward.argmax(1) == mb_best).float().mean()
                    reward_preferred = predicted_reward.gather(1, mb_best.view(-1, 1)).view(-1)
                    reward_rejected = predicted_reward.gather(1, (1 - mb_best).view(-1, 1)).view(-1)
                    if args.logsigmoid:
                        loss = -F.logsigmoid(reward_preferred - reward_rejected).mean()
                    else:
                        loss = F.cross_entropy(predicted_reward, mb_best)
                    accelerator.backward(loss)

                    # for k, v in reward_model.named_parameters():
                    #     if v.requires_grad:
                    #         if v.grad is None:
                    #             print(f"found unused param: {k}")

                    optimizer.step()  # accelerate handles gradient accumulation automatically
                    optimizer.zero_grad()
                    scheduler.step()
                    losses[gradient_accumulation_step] = loss
                    accuracies[gradient_accumulation_step] = accuracy
                    reward_preferreds[gradient_accumulation_step] = reward_preferred.mean()
                    reward_rejecteds[gradient_accumulation_step] = reward_rejected.mean()
                gradient_accumulation_step += 1

            train_accuracy = accelerator.gather(accuracies).mean().item()
            writer.add_scalar("train/loss", accelerator.gather(losses).mean().item(), global_step)
            writer.add_scalar("train/accuracy", train_accuracy, global_step)
            writer.add_scalar("train/reward_preferred", accelerator.gather(reward_preferreds).mean().item(), global_step)
            writer.add_scalar("train/reward_rejected", accelerator.gather(reward_rejecteds).mean().item(), global_step)
            lr = scheduler.get_last_lr()
            writer.add_scalar("train/lr", np.array(lr).mean().item(), global_step)
            accelerator.print("train/accuracy", train_accuracy)

            # if args.print_sample_output_freq > 0 and global_step % args.print_sample_output_freq == 0:
            if global_step == args.num_updates - 1: # first and last update
                dev_validation_accuracy = evaluate(args, accelerator, device, reward_model, dev_validation_label)
                writer.add_scalar("dev_validation/accuracy", dev_validation_accuracy, global_step)
                accelerator.print("dev_validation/accuracy", dev_validation_accuracy, global_step)
                eval_validation_accuracy = evaluate(args, accelerator, device, reward_model, eval_validation_label)
                writer.add_scalar("eval_validation/accuracy", eval_validation_accuracy, global_step)
                accelerator.print("eval_validation/accuracy", eval_validation_accuracy, global_step)
                eval_validation_accuracy = evaluate(args, accelerator, device, reward_model, label)
                writer.add_scalar("train_full/accuracy", eval_validation_accuracy, global_step)
                accelerator.print("train_full/accuracy", eval_validation_accuracy, global_step)

    torch.cuda.empty_cache()
    if args.normalize_after:
        print("===Normalize reward model *after* training===")
        print(
            "before normalization. "
            + f"Gain: {accelerator.unwrap_model(reward_model).reward_gain.data}"
            + f" Bias: {accelerator.unwrap_model(reward_model).reward_bias.data}"
        )

        normalize(
            tokenizer,
            accelerator,
            device,
            reward_model,
            reward_model,
            dataloader,
            validation_dataloader,
        )
        print(
            "after normalization. "
            + f"Gain: {accelerator.unwrap_model(reward_model).reward_gain.data}"
            + f" Bias: {accelerator.unwrap_model(reward_model).reward_bias.data}"
        )

    # save model
    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        # torch.save(accelerator.unwrap_model(reward_model).state_dict(), args.save_path)
        accelerator.save_model(reward_model, args.save_path)

    if accelerator.is_main_process and args.track:
        wandb.finish()


if __name__ == "__main__":
    args = tyro.cli(Args)
    train(args)
