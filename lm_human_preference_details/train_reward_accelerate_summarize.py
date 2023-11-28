from collections import defaultdict
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
from accelerate.utils import DistributedDataParallelKwargs, gather_object
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
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, get_scheduler


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
    query_dataset: str = "vwxyzjn/summarize_from_feedback_tldr_3_filtered_oai_preprocessing"

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
    wandb_project_name: str = "tldr_summarize"
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
    dropout_layer_keys: List[str] = field(default_factory=lambda: ["attn_pdrop", "embd_pdrop", "resid_pdrop", "summary_first_dropout"])
    """Which layers to apply dropout to"""
    deepspeed: bool = False
    """Whether to use deepspeed to train the model"""
    label_dataset: str = "vwxyzjn/summarize_from_feedback_oai_preprocessing"
    """the name of the dataset to use for labels in `https://huggingface.co/datasets/vwxyzjn/lm-human-preferences`"""
    local_batch_size: int = 8
    """per rank batch size"""
    gradient_accumulation_steps: int = 1
    """gradient accumulation steps"""
    local_micro_batch_size: tyro.conf.Suppress[int] = None
    """per rank micro batch size"""
    local_eval_batch_size: int = 8
    """per rank eval batch size"""
    lr: float = 5e-6
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
    normalize_before: bool = False
    """Whether, before training, to normalize the rewards on the policy to the scales on the training buffer. (For comparisons, just use mean 0, var 1.)"""
    normalize_after: bool = False
    """Whether, after training, to normalize the rewards on the ref policy to mean 0, var 1 (so the KL coefficient always has the same meaning)."""
    print_sample_output_freq: int = 300
    """How often to print sample output"""
    sft_model_path: str = ""
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
    scheduler: str = "cosine"
    """Which scheduler to use"""
    warm_up_steps: int = 0
    """Number of warm up steps for the scheduler"""
    task: TaskHParams = field(default_factory=TaskHParams)
    labels: LabelHParams = field(default_factory=LabelHParams)


# taken from https://github.com/microsoft/DeepSpeedExamples/blob/737c6740bec38b77a24a59135b6481a53d566b38/applications/DeepSpeed-Chat/training/utils/model/model_utils.py#L20C1-L26C52
def configure_dropout(model_config, dropout_layer_keys, dropout):
    if dropout is not None:
        for key in dropout_layer_keys:
            if hasattr(model_config, key):
                print(f"Setting model_config.{key} to {dropout}")
                setattr(model_config, key, dropout)


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
        reward = self.scalar_head(output.hidden_states[-1])
        return reward


def left_padding_to_right_padding(tokens, pad_id):
    """Convert from left padding to right padding."""
    assert tokens.ndim == 2
    return torch.tensor(
        [[x for x in row if x != pad_id] + [pad_id] * (row == pad_id).sum() for row in tokens],
        device=tokens.device,
    )


def ceil_div(a, b):
    return (a - 1) // b + 1


def exact_div(a, b):
    q = a // b
    if a != q * b:
        raise ValueError(f"Inexact division: {a} / {b} = {a / b}")
    return q


def get_reward(reward_model, query_responses, tokenizer):
    attention_mask = query_responses != tokenizer.pad_token_id
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    reward_logits = reward_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True,
        output_hidden_states=True,
    )
    sequence_lengths = (
        torch.eq(query_responses, tokenizer.pad_token_id).long().argmax(-1) - 1).to(
        query_responses.device
    )
    # https://github.com/huggingface/transformers/blob/dc68a39c8111217683bf49a4912d0c9018bab33d/src/transformers/models/gpt2/modeling_gpt2.py#L1454
    return reward_logits[torch.arange(reward_logits.size(0), device=reward_logits.device), sequence_lengths]


def evaluate(args, accelerator, tokenizer, reward_model, dataloader):
    reward_model.eval()
    with torch.no_grad():
        items = defaultdict(list)
        for data in tqdm(dataloader):
            mb_query = data["query_token"]
            mb_responses = torch.cat([data[f"response0_token"].unsqueeze(1), data[f"response1_token"].unsqueeze(1)], dim=1)
            mb_best = data["choice"]
            mb_query_tiled = mb_query.unsqueeze(1).repeat(1, args.labels.num_labels, 1)
            query_responses = torch.cat([mb_query_tiled, mb_responses], dim=2).flatten(0, 1)
            query_responses = left_padding_to_right_padding(query_responses, tokenizer.pad_token_id)
            predicted_reward = get_reward(reward_model, query_responses, tokenizer)
            predicted_reward = predicted_reward.view(-1, args.labels.num_labels)
            accuracy = (predicted_reward.argmax(1) == mb_best).float()

            for k in data:
                data[k] = gather_object(data[k])
            for i in range(len(accuracy)):
                items["query"].append(tokenizer.decode(data["query_token"][i], skip_special_tokens=True))
                items["response0"].append(tokenizer.decode(data["response0_token"][i]))
                items["response1"].append(tokenizer.decode(data["response1_token"][i]))
                items["batch"].append(data["batch"][i])
                items["split"].append(data["split"][i])
                items["confidence"].append(data["extra.confidence"][i].item())
                items["choice"].append(data["choice"][i].item())
                items["policies"].append(data["policies"][i])
                items["response0_policy"].append(data["response0_policy"][i])
                items["response1_policy"].append(data["response1_policy"][i])
                items["accuracy"].append(accuracy[i].item())
    reward_model.train()
    return pd.DataFrame(items)


# def train(args: Args):
if __name__ == "__main__":
    args = tyro.cli(Args)
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
    args.micro_batch_size = int(args.local_micro_batch_size * args.world_size)
    args.num_updates = args.labels.num_train // args.local_batch_size

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
    model_config = AutoConfig.from_pretrained(args.base_model)
    configure_dropout(model_config, args.dropout_layer_keys, 0.0)  # disable dropout
    if accelerator.is_main_process:
        pprint(model_config)
    reward_model = AutoModelForCausalLMWithRewardHead(
        AutoModelForCausalLM.from_pretrained(
            args.base_model,
            config=model_config,
            trust_remote_code=True,
        )
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
    scheduler = get_scheduler(
        args.scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.warm_up_steps,
        num_training_steps=args.num_updates * args.num_epochs,
    )

    if args.deepspeed:
        deepspeed_states = AcceleratorState().deepspeed_plugin
        deepspeed_states.deepspeed_config["train_micro_batch_size_per_gpu"] = args.local_micro_batch_size

    label = load_dataset(args.label_dataset, "comparisons", split="train")
    label = label.shuffle(seed=local_seed)
    label = label.select(range(args.labels.num_train))
    label = label.with_format("torch", columns=["query_token", "choice", "response0_token", "response1_token", "batch", "split"])
    dataloader = DataLoader(label, batch_size=args.local_micro_batch_size)
    reward_model, optimizer, dataloader, scheduler = accelerator.prepare(reward_model, optimizer, dataloader, scheduler)
    validation_label = load_dataset(args.label_dataset, "comparisons", split="validation").flatten()
    validation_label = validation_label.with_format("torch", columns=["query_token", "choice", "response0_token", "response1_token", "batch", "split", "extra.confidence", "response0_policy", "response1_policy", "policies"])
    validation_dataloader = DataLoader(validation_label, batch_size=args.local_eval_batch_size)
    validation_dataloader = accelerator.prepare(validation_dataloader)

    accelerator.print("===training reward model===")
    losses = torch.zeros((args.gradient_accumulation_steps,), device=device)
    accuracies = torch.zeros((args.gradient_accumulation_steps,), device=device)
    reward_preferreds = torch.zeros((args.gradient_accumulation_steps,), device=device)
    reward_rejecteds = torch.zeros((args.gradient_accumulation_steps,), device=device)
    reward_model.train()
    gradient_accumulation_idx = 0
    global_step = 0
    update = 0
    for epoch in range(args.num_epochs):
        accelerator.print(f"epoch: {epoch}")
        for data in dataloader:
            update += 1
            global_step += args.micro_batch_size
            mb_query = data["query_token"]
            mb_responses = torch.cat([data[f"response0_token"].unsqueeze(1), data[f"response1_token"].unsqueeze(1)], dim=1)
            mb_best = data["choice"]
            mb_query_tiled = mb_query.unsqueeze(1).repeat(1, args.labels.num_labels, 1)
            query_responses = torch.cat([mb_query_tiled, mb_responses], dim=2).flatten(0, 1)
            query_responses = left_padding_to_right_padding(query_responses, tokenizer.pad_token_id)
            with accelerator.accumulate(reward_model):
                predicted_reward = get_reward(reward_model, query_responses, tokenizer)
                predicted_reward = predicted_reward.view(-1, args.labels.num_labels)
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
            losses[gradient_accumulation_idx] = loss
            accuracies[gradient_accumulation_idx] = accuracy
            reward_preferreds[gradient_accumulation_idx] = reward_preferred.mean()
            reward_rejecteds[gradient_accumulation_idx] = reward_rejected.mean()
            gradient_accumulation_idx = (gradient_accumulation_idx + 1) % args.gradient_accumulation_steps
            if update > 1 and (update - 1) % args.gradient_accumulation_steps == 0:
                train_accuracy = accelerator.gather(accuracies).mean().item()
                writer.add_scalar("train/loss", accelerator.gather(losses).mean().item(), global_step)
                writer.add_scalar("train/accuracy", train_accuracy, global_step)
                writer.add_scalar("train/reward_preferred", accelerator.gather(reward_preferreds).mean().item(), global_step)
                writer.add_scalar("train/reward_rejected", accelerator.gather(reward_rejecteds).mean().item(), global_step)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
                accelerator.print(f"{train_accuracy=}, {scheduler.get_last_lr()=}, {update=}")
        # if args.print_sample_output_freq > 0 and global_step % args.print_sample_output_freq == 0:

        evaluate_df = evaluate(args, accelerator, tokenizer, reward_model, validation_dataloader)
        for split, row in evaluate_df[["split", "accuracy"]].groupby(["split"]).mean().iterrows():
            writer.add_scalar(f"eval/accuracy/{split}", row["accuracy"], global_step)
            accelerator.print(f"{split} accuracy: {row['accuracy']}")
        for batch, row in evaluate_df[["batch", "accuracy"]].groupby(["batch"]).mean().iterrows():
            writer.add_scalar(f"eval/accuracy/{batch}", row["accuracy"], global_step)
            accelerator.print(f"{batch} accuracy: {row['accuracy']}")
        for confi, row in evaluate_df[["confidence", "accuracy"]].groupby(["confidence"]).mean().iterrows():
            writer.add_scalar(f"eval/confidence/{confi}", row["accuracy"], global_step)
            accelerator.print(f"{confi} confidence: {row['accuracy']}")
        writer.add_scalar("eval/accuracy", evaluate_df["accuracy"].mean(), global_step)
        accelerator.print(f"eval accuracy: {evaluate_df['accuracy'].mean()}")
        if accelerator.is_main_process:
            os.makedirs(f"eval_tables/{run_name}", exist_ok=True)
            evaluate_df.to_csv(f"eval_tables/{run_name}/eval_{update}.csv")
            if args.track:
                wandb.log({"samples/query_responses": wandb.Table(dataframe=evaluate_df)}, step=update)


    torch.cuda.empty_cache()

    # save model
    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        accelerator.save_model(reward_model, args.save_path, max_shard_size="1000GB")

    if accelerator.is_main_process and args.track:
        wandb.finish()


if __name__ == "__main__":
    args = tyro.cli(Args)
    # train(args)
