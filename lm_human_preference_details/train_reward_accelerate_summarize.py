import os
import random
import time
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace
from typing import List, Optional

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
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from lm_human_preference_details.data import process_query


@dataclass
class LabelHParams:
    type: str = None
    num_train: int = 64832
    num_labels: int = 2
    source: str = None


@dataclass
class TaskHParams:
    # Query params
    query_length: int = 512
    query_dataset: str = "tldr_3_filtered"

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
    print_sample_output_freq: int = 60
    """How often to print sample output"""
    save_path: str = "models/reward.pt"
    """Where to save the model"""
    use_tensorflow_adam: bool = True
    """Whether to use tensorflow-style Adam optimizer instead of PyTorch's"""
    task: TaskHParams = field(default_factory=TaskHParams)
    labels: LabelHParams = field(default_factory=LabelHParams)


OPENAI_PAD_TOKEN_ID = 50259


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
        self.reward_gain = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.reward_bias = torch.nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def forward(self, **kwargs):
        output = self.lm_backbone(**kwargs)
        reward_latents = output.hidden_states[-1]
        # shape: [batch_size, length, hidden_size]
        last_reward_latents = reward_latents
        # shape: [batch_size, hidden_size]
        reward = self.scalar_head(last_reward_latents)
        # shape: [batch_size, 1]
        reward = self.reward_gain * reward + self.reward_bias
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


def generate(lm_backbone, queries, args, generation_config):
    """generate in a way that does not affect padding tokens"""
    context_length = queries.shape[1]
    attention_mask = queries != args.pad_token_id
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


def get_reward(reward_model, query_responses, args):
    attention_mask = query_responses != args.pad_token_id
    position_ids = attention_mask.cumsum(1) - attention_mask.long()  # exclusive cumsum
    input_ids = query_responses.clone()
    input_ids[~attention_mask] = 0
    return reward_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        return_dict=True,
        output_hidden_states=True,
    )


def normalize(
    args,
    accelerator,
    device,
    lm_backbone,
    reward_model,
    iter_dataloader,
    generation_config,
):
    with torch.no_grad():
        # reset reward scales
        accelerator.unwrap_model(reward_model).reward_gain.data.fill_(1.0)
        accelerator.unwrap_model(reward_model).reward_bias.data.fill_(0.0)
        # number of minibatches for computing the normalization statistics
        n_batches = ceil_div(args.local_normalize_samples, args.local_rollout_batch_size)
        sample_queries_responses = []
        for _ in range(n_batches):
            data = next(iter_dataloader)
            queries = data["query_token"].to(device)
            queries = right_padding_to_left_padding(data["query_token"], args.pad_token_id).to(device)
            query_responses = generate(lm_backbone, queries, args, generation_config)
            sample_queries_responses.append(query_responses)

        # compute reward statistics
        rewards = []
        for query_responses in sample_queries_responses:
            rewards.append(get_reward(reward_model, query_responses, args)[1])
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
        n_batches = ceil_div(args.local_normalize_samples, args.local_rollout_batch_size)
        sample_queries_responses = []
        for _ in range(n_batches):
            data = next(iter_dataloader)
            queries = data["query_token"].to(device)
            queries = right_padding_to_left_padding(data["query_token"], args.pad_token_id).to(device)
            query_responses = generate(lm_backbone, queries, args, generation_config)
            sample_queries_responses.append(query_responses)
        rewards = []
        for query_responses in sample_queries_responses:
            rewards.append(get_reward(reward_model, query_responses, args)[1])
        rewards = torch.cat(rewards)
        rewards = accelerator.gather(rewards)
        mean, std = rewards.mean(), rewards.std()
        print(f"after mean: {mean}, after std: {std}")


def train(args: Args):
    accelerator = Accelerator(
        kwargs_handlers=[
            DistributedDataParallelKwargs(
                broadcast_buffers=False,
            )
        ],  # this is needed to avoid https://github.com/pytorch/pytorch/issues/22095#issuecomment-505099500
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    args.world_size = accelerator.num_processes
    args.batch_size = int(args.local_batch_size * args.world_size)
    args.rollout_batch_size = int(args.local_rollout_batch_size * args.world_size)
    args.local_micro_batch_size = exact_div(args.local_batch_size, args.gradient_accumulation_steps)
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
    untrained_model = AutoModelForCausalLMWithRewardHead(
        AutoModelForCausalLM.from_pretrained(args.base_model, trust_remote_code=True)
    )
    reward_model = AutoModelForCausalLMWithRewardHead(
        AutoModelForCausalLM.from_pretrained(args.base_model, trust_remote_code=True)
    )
    untrained_model.lm_backbone.generation_config.eos_token_id = (
        None  # disable `pad_token_id` and `eos_token_id` because we just want to
    )
    untrained_model.lm_backbone.generation_config.pad_token_id = None  # generate tokens without truncation / padding
    reward_model.lm_backbone.generation_config.eos_token_id = (
        None  # disable `pad_token_id` and `eos_token_id` because we just want to
    )
    reward_model.lm_backbone.generation_config.pad_token_id = None  # generate tokens without truncation / padding
    # make sure the `lm_head` or `embed_out` does not require gradients, otherwise
    # pytorch DDP complains; see https://gist.github.com/vwxyzjn/45fc8706dfb3cf33695f0f57cc44a533
    if isinstance(reward_model.lm_backbone, transformers.GPTNeoXForCausalLM):
        reward_model.lm_backbone.embed_out.requires_grad_(False)
    if args.use_tensorflow_adam:
        optimizer = AdamTensorFlowStyle(reward_model.parameters(), lr=args.lr, eps=args.eps)
    else:
        optimizer = optim.Adam(reward_model.parameters(), lr=args.lr, eps=args.eps)
    dataset = load_dataset("vwxyzjn/summarize_from_feedback_tldr_3_filtered", split="train")

    def process_query_data(x):
        return {
            **process_query(x, encoder=tokenizer, hparams=patch_h),
        }

    dataset = dataset.map(process_query_data)
    dataset = dataset.with_format("torch", columns=["query_token"])
    dataset = dataset.shuffle(seed=local_seed)
    dataloader = DataLoader(dataset, batch_size=args.local_rollout_batch_size)
    reward_model, optimizer, dataloader = accelerator.prepare(reward_model, optimizer, dataloader)
    if args.deepspeed:
        import deepspeed

        deepspeed_states = AcceleratorState().deepspeed_plugin
        # deepspeed_states.deepspeed_config["train_micro_batch_size_per_gpu"] = args.ppo.local_micro_batch_size
        # deepspeed_states.deepspeed_config["checkpoint"] = {"use_node_local_storage": True}
        eval_ds_config = {
            "train_micro_batch_size_per_gpu": deepspeed_states.deepspeed_config["train_micro_batch_size_per_gpu"],
            # "steps_per_print": 10,
            # "zero_optimization": {
            #     "stage": stage,
            #     "stage3_param_persistence_threshold": 1e4,
            #     "offload_param": {
            #         "device": off_load_device
            #     }
            # },
            "bf16": {"enabled": True},
            "prescale_gradients": False,
            "wall_clock_breakdown": False,
        }
        untrained_model, *_ = deepspeed.initialize(model=untrained_model, config=eval_ds_config)
        untrained_model.eval()
    else:
        untrained_model = untrained_model.to(device)

    def repeat_generator():  # TODO: ideally we shuffle the dataloader as well
        while True:
            yield from dataloader

    iter_dataloader = iter(repeat_generator())
    generation_config = GenerationConfig(
        max_new_tokens=args.task.response_length,
        min_new_tokens=args.task.response_length,
        temperature=args.task.temperature,
        top_k=0.0,
        top_p=1.0,
        do_sample=True,
    )

    if args.normalize_before:
        print("===Normalize reward model *before* training===")
        print(
            "before normalization. "
            + f"Gain: {accelerator.unwrap_model(reward_model).reward_gain.data}"
            + f" Bias: {accelerator.unwrap_model(reward_model).reward_bias.data}"
        )

        normalize(
            args,
            accelerator,
            device,
            untrained_model.lm_backbone,
            reward_model,
            iter_dataloader,
            generation_config,
        )
        print(
            "after normalization. "
            + f"Gain: {accelerator.unwrap_model(reward_model).reward_gain.data}"
            + f" Bias: {accelerator.unwrap_model(reward_model).reward_bias.data}"
        )

    # `label` has keys `['sample0', 'query', 'best', 'sample3', 'sample1', 'sample2']`
    label = load_dataset(args.label_dataset, "comparisons", split="train")
    test_label = load_dataset(args.label_dataset, "comparisons", split="validation")
    print("Num labels found in source:", len(label))
    print("training on", args.labels.num_train, "in batches of", args.local_batch_size)

    def process_response_data(x):
        return {
            **process_query(x["info"], encoder=tokenizer, hparams=patch_h),
            "response0_token": tokenizer.encode(
                x["summaries"][0]["text"], padding="max_length", max_length=args.task.response_length, truncation=True
            ),
            "response1_token": tokenizer.encode(
                x["summaries"][1]["text"], padding="max_length", max_length=args.task.response_length, truncation=True
            ),
        }

    label = label.map(process_response_data)
    test_label = test_label.map(process_response_data)
    #  tokenizer.encode(label[0]["summaries"][0]["text"])

    print("===training reward model===")
    all_inds = np.random.permutation(args.labels.num_train)
    # ensure that all processes have the same shuffled indices
    all_inds = broadcast(torch.tensor(all_inds, device=device), 0)
    all_inds = all_inds.cpu().numpy()
    global_step = 0
    for start in range(0, args.labels.num_train, args.batch_size):
        # linear rate annealing
        lr = (1 - start / args.labels.num_train) * args.lr
        optimizer.param_groups[0]["lr"] = lr

        global_step += 1
        end = start + args.batch_size
        b_inds_all = all_inds[start:end]
        b_inds = b_inds_all[accelerator.process_index :: accelerator.num_processes]  #  multi-GPU slicing
        losses = torch.zeros((args.gradient_accumulation_steps,), device=device)
        accuracies = torch.zeros((args.gradient_accumulation_steps,), device=device)
        gradient_accumulation_step = 0
        for micro_batch_start in range(0, args.local_batch_size, args.local_micro_batch_size):
            with accelerator.accumulate(reward_model):
                micro_batch_end = micro_batch_start + args.local_micro_batch_size
                micro_batch_inds = b_inds[micro_batch_start:micro_batch_end]
                mb_data = label[micro_batch_inds]
                mb_query = torch.from_numpy(np.stack(mb_data["query_token"])).to(device)
                mb_best = torch.from_numpy(np.stack(mb_data["choice"])).to(device)
                mb_responses = [
                    torch.from_numpy(np.stack(mb_data[f"response{i}_token"])).to(device) for i in range(args.labels.num_labels)
                ]
                predicted_rewards = []
                for i in range(args.labels.num_labels):
                    query_responses = torch.cat([mb_query, mb_responses[i]], dim=1)
                    reward = get_reward(reward_model, query_responses, args)[1]
                    last_response_indices = first_true_indices(query_responses == args.pad_token_id) - 1
                    last_response_indices = torch.max(
                        last_response_indices,
                        torch.zeros([1], dtype=last_response_indices.dtype, device=query_responses.device),
                    )
                    predicted_rewards.append(reward[:, :, 0].gather(1, last_response_indices.unsqueeze(1)).view(-1))
                predicted_rewards = torch.stack(
                    predicted_rewards, dim=1
                )  # shape (batch_size, num_labels), basically a reward prediction for each label
                reward_preferred = predicted_rewards.gather(1, mb_best.view(-1, 1)).view(-1)
                reward_rejected = predicted_rewards.gather(1, (1 - mb_best).view(-1, 1)).view(-1)
                accuracy = (predicted_rewards.argmax(1) == mb_best).float().mean()
                loss = -nn.functional.logsigmoid(reward_preferred - reward_rejected).mean()
                # loss = torch.nn.functional.cross_entropy(predicted_rewards, mb_best)
                accelerator.backward(loss)
                optimizer.step()  # accelerate handles gradient accumulation automatically
                optimizer.zero_grad()
                losses[gradient_accumulation_step] = loss
                accuracies[gradient_accumulation_step] = accuracy
            gradient_accumulation_step += 1

        train_accuracy = accelerator.gather(accuracies).mean().item()
        writer.add_scalar("train/loss", accelerator.gather(losses).mean().item(), global_step)
        writer.add_scalar("train/accuracy", train_accuracy, global_step)
        writer.add_scalar("train/lr", lr, global_step)
        print("train/accuracy", train_accuracy)

        if args.print_sample_output_freq > 0 and global_step % args.print_sample_output_freq == 0:
            with torch.no_grad():
                # eval on test_label, some duplicate code (I don't want to make the training loop into a function...)
                test_accuracies = []
                eval_len = 200 # len(test_label)
                len_labels = (eval_len // args.batch_size) * args.batch_size  # in case the last batch is not full
                new_all_inds = np.arange(len_labels)
                for start in range(0, len_labels, args.batch_size):
                    end = start + args.batch_size
                    b_inds_all = new_all_inds[start:end]
                    b_inds = b_inds_all[accelerator.process_index :: accelerator.num_processes]  #  multi-GPU slicing
                    for micro_batch_start in range(0, args.local_batch_size, args.local_micro_batch_size):
                        micro_batch_end = micro_batch_start + args.local_micro_batch_size
                        micro_batch_inds = b_inds[micro_batch_start:micro_batch_end]
                        mb_data = label[micro_batch_inds]
                        mb_query = torch.from_numpy(np.stack(mb_data["query_token"])).to(device)
                        mb_best = torch.from_numpy(np.stack(mb_data["choice"])).to(device)
                        mb_responses = [
                            torch.from_numpy(np.stack(mb_data[f"response{i}_token"])).to(device)
                            for i in range(args.labels.num_labels)
                        ]
                        predicted_rewards = []
                        for i in range(args.labels.num_labels):
                            query_responses = torch.cat([mb_query, mb_responses[i]], dim=1)
                            reward = get_reward(reward_model, query_responses, args)[1]
                            last_response_indices = first_true_indices(query_responses == args.pad_token_id) - 1
                            last_response_indices = torch.max(
                                last_response_indices,
                                torch.zeros([1], dtype=last_response_indices.dtype, device=query_responses.device),
                            )
                            predicted_rewards.append(reward[:, :, 0].gather(1, last_response_indices.unsqueeze(1)).view(-1))
                        predicted_rewards = torch.stack(
                            predicted_rewards, dim=1
                        )  # shape (batch_size, num_labels), basically a reward prediction for each label
                        reward_preferred = predicted_rewards.gather(1, mb_best.view(-1, 1)).view(-1)
                        reward_rejected = predicted_rewards.gather(1, (1 - mb_best).view(-1, 1)).view(-1)
                        accuracy = (predicted_rewards.argmax(1) == mb_best).float().mean()
                        test_accuracies.append(accuracy)
                test_accuracy = accelerator.gather(torch.stack(test_accuracies).mean()).mean().item()
                writer.add_scalar("test/accuracy", test_accuracy, global_step)
                if accelerator.is_main_process:
                    print("test/accuracy", test_accuracy, global_step)

                # the part below is testing out some generations and KLs, not presented in the original code
                data = next(iter_dataloader)
                queries = data["query_token"].to(device)
                context_length = queries.shape[1]
                queries = right_padding_to_left_padding(data["query_token"], args.pad_token_id).to(device)
                query_responses = generate(
                    accelerator.unwrap_model(reward_model).lm_backbone,
                    queries,
                    args,
                    generation_config,
                )
                responses = query_responses[:, context_length:]

                output, reward = get_reward(reward_model, query_responses, args)
                logits = output.logits[:, context_length - 1 : -1]
                logits /= args.task.temperature
                all_logprobs = F.log_softmax(logits, dim=-1)
                logprobs = torch.gather(all_logprobs, 2, responses.unsqueeze(-1)).squeeze(-1)
                del output, logits, all_logprobs
                torch.cuda.empty_cache()

                output, _ = get_reward(untrained_model, query_responses, args)
                logits = output.logits[:, context_length - 1 : -1]
                logits /= args.task.temperature
                all_logprobs = F.log_softmax(logits, dim=-1)
                ref_logprobs = torch.gather(all_logprobs, 2, responses.unsqueeze(-1)).squeeze(-1)
                del output, logits, all_logprobs
                torch.cuda.empty_cache()

                kl = logprobs - ref_logprobs
                kl_sum = kl.sum(axis=1)
                all_decode_queries = tokenizer.batch_decode(queries, skip_special_tokens=True)
                all_query_responses = tokenizer.batch_decode(query_responses, skip_special_tokens=True)
                all_responses = [x[len(y) :] for x, y in zip(all_query_responses, all_decode_queries)]
                all_df = pd.DataFrame(
                    {
                        "query": all_decode_queries,
                        "response": all_responses,
                        "kl": kl_sum.float().cpu().numpy(),
                    }
                )
                if accelerator.is_main_process and args.track:
                    wandb.log({"query_responses": wandb.Table(dataframe=all_df)}, step=global_step)
                try:
                    print_rich_table(f"Sample Output at Step {global_step}", all_df[:4], console)
                except Exception as e:
                    print(e)
                    pass
                del (
                    query_responses,
                    all_decode_queries,
                    all_query_responses,
                    all_responses,
                    kl_sum,
                    all_df,
                )
                writer.add_scalar("train/kl", kl.sum(1).mean().item(), global_step)

    torch.cuda.empty_cache()
    if args.normalize_after:
        print("===Normalize reward model *after* training===")
        print(
            "before normalization. "
            + f"Gain: {accelerator.unwrap_model(reward_model).reward_gain.data}"
            + f" Bias: {accelerator.unwrap_model(reward_model).reward_bias.data}"
        )

        normalize(
            args,
            accelerator,
            device,
            untrained_model.lm_backbone,
            reward_model,
            iter_dataloader,
            generation_config,
        )
        print(
            "after normalization. "
            + f"Gain: {accelerator.unwrap_model(reward_model).reward_gain.data}"
            + f" Bias: {accelerator.unwrap_model(reward_model).reward_bias.data}"
        )

    # save model
    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        torch.save(accelerator.unwrap_model(reward_model).state_dict(), args.save_path)

    if accelerator.is_main_process and args.track:
        wandb.finish()


if __name__ == "__main__":
    args = tyro.cli(Args)
    train(args)
