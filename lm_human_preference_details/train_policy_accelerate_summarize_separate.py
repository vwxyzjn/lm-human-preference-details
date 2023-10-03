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
import tyro
from accelerate import Accelerator
from accelerate.state import AcceleratorState
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

INVALID_LOGPROB = 1.0

@dataclass
class AdaptiveKLParams:
    target: float = 6.0
    horizon: int = 10000  # in episodes


@dataclass
class RewardHParams:
    kl_coef: float = 0.15
    use_adaptive_kl: bool = True
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
    gradient_accumulation_steps: int = 1
    """gradient accumulation steps"""
    local_micro_batch_size: tyro.conf.Suppress[int] = None
    """per rank micro batch size"""
    world_size: tyro.conf.Suppress[int] = None
    batch_size: tyro.conf.Suppress[int] = None
    minibatch_size: tyro.conf.Suppress[int] = None
    num_updates: tyro.conf.Suppress[int] = None
    nminibatches: int = 1
    noptepochs: int = 4
    lr: float = 0.00001
    eps: float = 1e-5
    vf_coef: float = 0.1
    cliprange: float = 0.2
    cliprange_value: float = 0.2
    gamma: float = 1
    lam: float = 0.95
    whiten_rewards: bool = True


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
    print_sample_output_freq: int = 10
    """How often to print sample output"""
    sft_model_path: str = "models/sft_policy.pt"
    """Where to load the SFT model"""
    save_path: str = "models/policy.pt"
    """Where to save the model"""
    use_tensorflow_adam: bool = True
    """Whether to use tensorflow-style Adam optimizer instead of PyTorch's"""
    task: TaskHParams = field(default_factory=TaskHParams)
    rewards: RewardHParams = field(default_factory=RewardHParams)
    ppo: PpoHParams = field(default_factory=PpoHParams)


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


# taken from https://github.com/OpenLMLab/MOSS-RLHF/blob/40b91eb2f2b71b16919addede0341d2bef70825d/ppo/ppo_trainer.py#L29
# we did this we can do a single `model = accelerator.prepare(model)`
class PolicyAndValueWrapper(nn.Module):
    def __init__(self, policy, critic) -> None:
        super().__init__()
        self.policy = policy
        self.critic = critic
    
    def forward(self, **kwargs):
        return self.policy(**kwargs), self.critic(**kwargs)
    

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


def get_reward_complete(reward_model, query_responses, args):
    reward = get_reward(reward_model, query_responses, args)[1]
    last_response_indices = first_true_indices(query_responses == args.pad_token_id) - 1
    last_response_indices = torch.max(
        last_response_indices,
        torch.zeros([1], dtype=last_response_indices.dtype, device=query_responses.device),
    )
    return reward[:, :, 0].gather(1, last_response_indices.unsqueeze(1)).view(-1)


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
        # accelerator.unwrap_model(reward_model).reward_gain.data.fill_(1.0)
        # accelerator.unwrap_model(reward_model).reward_bias.data.fill_(0.0)
        # number of minibatches for computing the normalization statistics
        rewards = []
        for data in dataloader:
            idx += len(data["query_token"])
            queries = data["query_token"].to(device)
            queries = right_padding_to_left_padding(queries, tokenizer.pad_token_id).to(device)
            reference_response = data["reference_response"].to(device)
            query_responses = torch.cat((queries, reference_response), dim=1)
            score = get_reward_complete(reward_model, query_responses, tokenizer)
            accelerator.print(score.shape, accelerator.gather(score).mean())
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


def forward(policy, query_responses, tokenizer):
    attention_mask = query_responses != tokenizer.pad_token_id
    position_ids = attention_mask.cumsum(1) - attention_mask.long()  # exclusive cumsum
    input_ids = query_responses.clone()
    input_ids[~attention_mask] = 0
    return policy(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        return_dict=True,
        output_hidden_states=True,
    )


# def train(args: Args):
if __name__ == "__main__":
    args = tyro.cli(Args)
    accelerator = Accelerator(gradient_accumulation_steps=args.ppo.gradient_accumulation_steps)
    args.ppo.world_size = accelerator.num_processes
    args.ppo.batch_size = int(args.ppo.local_batch_size * args.ppo.world_size)
    args.ppo.minibatch_size = exact_div(args.ppo.batch_size, args.ppo.nminibatches)
    args.ppo.local_mini_batch_size = exact_div(args.ppo.local_batch_size, args.ppo.nminibatches)
    args.ppo.local_micro_batch_size = exact_div(args.ppo.local_mini_batch_size, args.ppo.gradient_accumulation_steps)
    patch_h = TaskQueryHParams(
        length=args.task.query_length,
        dataset=args.task.query_dataset,
        format_str=args.task.query_format_str,
        truncate_field=args.task.query_truncate_field,
        truncate_text=args.task.query_truncate_text,
        padding=args.task.query_padding,
        pad_side=args.task.query_pad_side,
    )
    if args.ppo.whiten_rewards:
        assert (
            args.ppo.local_mini_batch_size >= 8
        ), f"Per-rank minibatch size {args.ppo.local_mini_batch_size} is insufficient for whitening"
    # `per_rank_rollout_batch_size` is our `args.ppo.local_batch_size`
    # `per_rank_minibatch_size` is our `args.ppo.local_mini_batch_size`
    args.ppo.num_updates = args.ppo.total_episodes // args.ppo.batch_size

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
    reward_model = AutoModelForCausalLMWithRewardHead(
        AutoModelForCausalLM.from_pretrained(args.base_model, trust_remote_code=True)
    )
    critic = AutoModelForCausalLMWithRewardHead(
        AutoModelForCausalLM.from_pretrained(args.base_model, trust_remote_code=True)
    )
    if args.rewards.trained_model:
        reward_model.load_state_dict(torch.load(args.rewards.trained_model, map_location=device))
        critic.load_state_dict(torch.load(args.rewards.trained_model, map_location=device))
        print(f"loaded pretrained reward model from {args.rewards.trained_model}")
    # each class should have a separate pretrained model that do not share weights
    ref_policy = AutoModelForCausalLM.from_pretrained(args.base_model, trust_remote_code=True)
    policy = AutoModelForCausalLM.from_pretrained(args.base_model, trust_remote_code=True)
    if args.sft_model_path:
        policy.load_state_dict(torch.load(args.sft_model_path, map_location=device))
        ref_policy.load_state_dict(torch.load(args.sft_model_path, map_location=device))
        print(f"loaded pretrained policy from {args.sft_model_path}")
    policy.generation_config.eos_token_id = (
        None  # disable `pad_token_id` and `eos_token_id` because we just want to
    )
    policy.generation_config.pad_token_id = None  # generate tokens without truncation / padding
    model = PolicyAndValueWrapper(policy, critic)
    if args.use_tensorflow_adam:
        optimizer = AdamTensorFlowStyle(model.parameters(), lr=args.ppo.lr, eps=args.ppo.eps)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.ppo.lr, eps=args.ppo.eps)
    dataset = load_dataset(args.task.query_dataset, split="train")
    validation_dataset = load_dataset(args.task.query_dataset, split="validation")

    def process_query_data(x):
        return {
            **process_query(x, encoder=tokenizer, hparams=patch_h),
            "reference_response": tokenizer.encode(
                f" {x['summary']}", padding="max_length", max_length=args.task.response_length, truncation=True,
                # with an extra leading space to account for the space between the query and response
            ),
        }

    dataset = dataset.map(process_query_data)
    dataset = dataset.with_format("torch", columns=["query_token", "reference_response"])
    dataset = dataset.shuffle(seed=local_seed)
    dataloader = DataLoader(dataset, batch_size=args.ppo.local_batch_size)
    validation_dataset = validation_dataset.map(process_query_data)
    validation_dataset = validation_dataset.with_format("torch", columns=["query_token", "reference_response"])
    validation_dataset = validation_dataset.shuffle(seed=local_seed)
    validation_dataloader = DataLoader(validation_dataset, batch_size=args.ppo.local_batch_size)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
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
        reward_model, *_ = deepspeed.initialize(model=reward_model, config=eval_ds_config)
        reward_model.eval()
        ref_policy, *_ = deepspeed.initialize(model=ref_policy, config=eval_ds_config)
        ref_policy.eval()
    else:
        ref_policy = ref_policy.to(device)
        reward_model = reward_model.to(device)

    def repeat_generator():  # TODO: ideally we shuffle the dataloader as well
        while True:
            yield from dataloader

    iter_dataloader = iter(repeat_generator())
    kl_ctl = AdaptiveKLController(args.rewards.kl_coef, hparams=args.rewards.adaptive_kl)
    # WARNING: even with `max_new_tokens` and `min_new_tokens` set to the same value, the number of tokens generated
    # may not be the same. TODO: investigate further, we just want to generate a fixed number of tokens
    generation_config = GenerationConfig(
        max_new_tokens=args.task.response_length,
        min_new_tokens=args.task.response_length,
        temperature=(args.task.temperature + 1e-7),
        top_k=0.0,
        top_p=1.0,
        do_sample=True,
    )

    # print("===Normalize reward model *before* training===")
    # print(
    #     "before normalization. "
    #     + f"Gain: {accelerator.unwrap_model(reward_model).reward_gain.data}"
    #     + f" Bias: {accelerator.unwrap_model(reward_model).reward_bias.data}"
    # )

    # normalize(
    #     tokenizer,
    #     accelerator,
    #     device,
    #     reward_model,
    #     reward_model,
    #     dataloader,
    #     validation_dataloader,
    # )
    # print(
    #     "after normalization. "
    #     + f"Gain: {accelerator.unwrap_model(reward_model).reward_gain.data}"
    #     + f" Bias: {accelerator.unwrap_model(reward_model).reward_bias.data}"
    # )
    # # # save model
    # # if args.save_path:
    # #     os.makedirs(os.path.dirname("models/correct_reward.pt"), exist_ok=True)
    # #     torch.save(accelerator.unwrap_model(reward_model).state_dict(), "models/correct_reward.pt")
    # raise

    print("===training policy===")
    global_step = 0
    stats_shape = (args.ppo.noptepochs, args.ppo.nminibatches, args.ppo.gradient_accumulation_steps)
    approxkls_stats = torch.zeros(stats_shape, device=device)
    clipfracs_stats = torch.zeros(stats_shape, device=device)
    pg_losses_stats = torch.zeros(stats_shape, device=device)
    vf_losses_stats = torch.zeros(stats_shape, device=device)
    vf_clipfrac_stats = torch.zeros(stats_shape, device=device)
    entropies_stats = torch.zeros(stats_shape, device=device)
    for update in range(1, args.ppo.num_updates + 1):
        global_step += 1 * args.ppo.batch_size
        frac = 1.0 - (update - 1.0) / args.ppo.num_updates
        lrnow = frac * args.ppo.lr
        optimizer.param_groups[0]["lr"] = lrnow
        data = next(iter_dataloader)
        with torch.no_grad():
            """
            let's use `P` to denote the padding token, `T` to denote the truncate token, and `X` to denote the
            actual tokens.
            queries: `PPXXX`
            query_responses: `PPXXX,XXXXTXX` # the space separates the query and response
            response: `XXXXTXX`
            postprocessed_responses: `XXXXTXX` -> `XXXXTPP`
            postprocessed_query_responses: `PPXXX,XXXXTPP`
            scores:                                  ↑    # corresponding to this `X` token
            
            """
            queries = data["query_token"].to(device)
            reference_responses = data["reference_response"].to(device)
            queries = right_padding_to_left_padding(data["query_token"], tokenizer.pad_token_id).to(device)
            query_reference_responses = torch.cat((queries, reference_responses), dim=1)



            reference_scores = get_reward_complete(reward_model, query_reference_responses, tokenizer)
            accelerator.print(accelerator.gather(reference_scores).mean())


            query_responses = generate(
                accelerator.unwrap_model(model).policy,
                queries,
                tokenizer,
                generation_config,
            )
            context_length = queries.shape[1]
            responses = query_responses[:, context_length:]

            output = forward(accelerator.unwrap_model(model).policy, query_responses, tokenizer)
            full_values = get_reward(accelerator.unwrap_model(model).critic, query_responses, tokenizer)[1]
            values = full_values[:, context_length - 1 : -1].squeeze(-1)
            logits = output.logits[:, context_length - 1 : -1]
            logits /= (args.task.temperature + 1e-7)
            all_logprobs = F.log_softmax(logits, dim=-1)
            logprobs = torch.gather(all_logprobs, 2, responses.unsqueeze(-1)).squeeze(-1)
            del output, logits, all_logprobs
            torch.cuda.empty_cache()

            ref_output = forward(ref_policy, query_responses, tokenizer)
            ref_logits = ref_output.logits[:, context_length - 1 : -1]
            ref_logits /= (args.task.temperature + 1e-7)
            ref_all_logprobs = F.log_softmax(ref_logits, dim=-1)
            ref_logprobs = torch.gather(ref_all_logprobs, 2, responses.unsqueeze(-1)).squeeze(-1)
            del ref_output, ref_logits, ref_all_logprobs
            torch.cuda.empty_cache()

            # **Response Processing**
            # 1. truncate at the first occurrence of `truncate_token` that appears at or after
            # position truncate_after in the responses
            # https://github.com/openai/lm-human-preferences/blob/cbfd210bb8b08f6bc5c26878c10984b90f516c66/lm_human_preferences/train_policy.py#L378
            truncate_token_mask = responses == args.task.truncate_token
            truncate_after_or_token_mask = torch.cat(
                [
                    torch.zeros_like(truncate_token_mask)[:, : args.task.truncate_after],
                    truncate_token_mask[:, args.task.truncate_after :],
                ],
                dim=1,
            )
            truncate_mask = (torch.cumsum(truncate_after_or_token_mask, dim=1) - truncate_after_or_token_mask.long()).bool()
            postprocessed_responses = torch.where(
                truncate_mask,
                torch.full_like(responses, tokenizer.pad_token_id),
                responses,
            )
            del truncate_token_mask, truncate_after_or_token_mask, truncate_mask
            torch.cuda.empty_cache()

            # 2. run reward model on the truncated responses
            postprocessed_query_responses = torch.cat((queries, postprocessed_responses), 1)
            padding_mask = (postprocessed_query_responses == tokenizer.pad_token_id)[:, context_length - 1 : -1]
            logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
            ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)
            values = torch.masked_fill(values, padding_mask, 0)

            scores = get_reward_complete(reward_model, postprocessed_query_responses, tokenizer)
            reference_scores = get_reward_complete(reward_model, query_reference_responses, tokenizer)

            # 3. filter response. Ensure that the sample contains truncate_token
            # responses not passing that filter will receive a low (fixed) score
            # only query humans on responses that pass that filter
            matches_token = postprocessed_responses[:, args.task.truncate_after :] == args.task.truncate_token
            filter_mask = torch.any(matches_token, dim=-1)
            scores = torch.where(
                filter_mask,
                scores,
                torch.full_like(scores, args.task.penalty_reward_value),
            )
            del matches_token, filter_mask
            torch.cuda.empty_cache()

            # 4. compute rewards
            kl = logprobs - ref_logprobs
            non_score_reward = -kl_ctl.value * kl
            rewards = non_score_reward.clone()
            rewards[:, -1] += scores

            # 5. whiten rewards
            if args.ppo.whiten_rewards:
                rewards = whiten(rewards, shift_mean=False)

            if args.print_sample_output_freq > 0 and (update - 1) % args.print_sample_output_freq == 0:
                try:
                    all_decode_queries = tokenizer.batch_decode(queries, skip_special_tokens=True)
                    all_postprocessed_query_responses = tokenizer.batch_decode(
                        postprocessed_query_responses, skip_special_tokens=True
                    )
                    all_postprocessed_responses = [
                        x[len(y) :] for x, y in zip(all_postprocessed_query_responses, all_decode_queries)
                    ]
                    all_reference_responses = tokenizer.batch_decode(reference_responses, skip_special_tokens=True)

                    kl_sum = kl.sum(axis=1)
                    all_df = pd.DataFrame(
                        {
                            "query": all_decode_queries,
                            "response": all_postprocessed_responses,
                            "reference_responses": all_reference_responses,
                            "score": scores.float().cpu().numpy(),
                            "reference_scores": reference_scores.float().cpu().numpy(),
                            "kl": kl_sum.float().cpu().numpy(),
                            "reward": (scores - kl_ctl.value * kl_sum).float().cpu().numpy(),
                        }
                    )
                    if accelerator.is_main_process and args.track:
                        wandb.log({"query_responses": wandb.Table(dataframe=all_df)}, step=update)
                    print_rich_table("stuff", all_df[:4], console)
                except Exception as e:
                    print(e)
                del (
                    all_decode_queries,
                    all_postprocessed_query_responses,
                    all_postprocessed_responses,
                    kl_sum,
                    all_df,
                )
            del postprocessed_query_responses
            torch.cuda.empty_cache()

            # 6. compute advantages and returns
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
            return_mean, return_var = returns.mean(), returns.var()
            value_mean, value_var = values.mean(), values.var()

        # Do multiple epochs of PPO training, with a fresh random shuffle in each epoch
        for ppo_epoch_idx in range(args.ppo.noptepochs):
            b_inds = np.random.permutation(args.ppo.local_batch_size)
            minibatch_idx = 0
            for mini_batch_start in range(0, args.ppo.local_batch_size, args.ppo.local_mini_batch_size):
                mini_batch_end = mini_batch_start + args.ppo.local_mini_batch_size
                mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
                gradient_accumulation_idx = 0
                for micro_batch_start in range(0, args.ppo.local_mini_batch_size, args.ppo.local_micro_batch_size):
                    with accelerator.accumulate(policy):
                        micro_batch_end = micro_batch_start + args.ppo.local_micro_batch_size
                        micro_batch_inds = mini_batch_inds[micro_batch_start:micro_batch_end]
                        mb_return = returns[micro_batch_inds]
                        mb_advantage = advantages[micro_batch_inds]
                        mb_values = values[micro_batch_inds]
                        mb_responses = responses[micro_batch_inds]
                        mb_query_responses = query_responses[micro_batch_inds]
                        mb_logprobs = logprobs[micro_batch_inds]

                        # output, vpred_temp = forward(policy, mb_query_responses, tokenizer)
                        output, (_, vpred_temp) = forward(model, mb_query_responses, tokenizer)
                        
                        logits = output.logits[:, context_length - 1 : -1]
                        logits /= (args.task.temperature + 1e-7)
                        new_all_logprobs = F.log_softmax(logits, dim=-1)
                        new_logprobs = torch.gather(new_all_logprobs, 2, mb_responses.unsqueeze(-1)).squeeze(-1)
                        new_logprobs = torch.masked_fill(new_logprobs, padding_mask[micro_batch_inds], INVALID_LOGPROB)
                        vpred = vpred_temp[:, context_length - 1 : -1].squeeze(-1)
                        vpred = torch.masked_fill(vpred, padding_mask[micro_batch_inds], 0)
                        vpredclipped = torch.clamp(
                            vpred,
                            mb_values - args.ppo.cliprange_value,
                            mb_values + args.ppo.cliprange_value,
                        )
                        vf_losses1 = torch.square(vpred - mb_return)
                        vf_losses2 = torch.square(vpredclipped - mb_return)
                        vf_loss = 0.5 * torch.max(vf_losses1, vf_losses2).mean()
                        vf_clipfrac = (vf_losses2 > vf_losses1).float().mean()
                        logprobs_diff = new_logprobs - mb_logprobs
                        ratio = torch.exp(logprobs_diff)
                        pg_losses = -mb_advantage * ratio
                        pg_losses2 = -mb_advantage * torch.clamp(ratio, 1.0 - args.ppo.cliprange, 1.0 + args.ppo.cliprange)
                        pg_loss = torch.max(pg_losses, pg_losses2).mean()
                        pg_clipfrac = (pg_losses2 > pg_losses).float().mean()
                        loss = pg_loss + args.ppo.vf_coef * vf_loss
                        accelerator.backward(loss)
                        optimizer.step()
                        optimizer.zero_grad()
                        prob_dist = torch.nn.functional.softmax(logits, dim=-1)
                        entropy = torch.logsumexp(logits, dim=-1) - torch.sum(prob_dist * logits, dim=-1)
                        approxkl = 0.5 * (logprobs_diff**2).mean()
                        with torch.no_grad():
                            approxkls_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = approxkl
                            clipfracs_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = pg_clipfrac
                            pg_losses_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = pg_loss
                            vf_losses_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = vf_loss
                            vf_clipfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = vf_clipfrac
                            entropies_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = entropy.mean()
                    gradient_accumulation_idx += 1
                minibatch_idx += 1
                if accelerator.is_main_process:
                    console.print(
                        f"ppo_epoch_idx",
                        ppo_epoch_idx,
                        "approxkl",
                        approxkl.item(),
                        "pg_loss",
                        pg_loss.item(),
                        "pg_clipfrac",
                        pg_clipfrac.item(),
                        "ratio",
                        ratio.mean().item(),
                    )

        with torch.no_grad():
            if not args.deepspeed:  # for some reason there is a OOM with the `writer.add_histogram`
                writer.add_histogram("ppo/val/ratio_hist", ratio, update)
            kl = logprobs - ref_logprobs
            mean_kl = kl.sum(1).mean()
            mean_entropy = (-logprobs).sum(1).mean()
            mean_non_score_reward = non_score_reward.sum(1).mean()
            writer.add_scalar("objective/kl_coef", kl_ctl.value, update)
            writer.add_scalar("objective/kl", accelerator.gather(mean_kl).mean().item(), update)
            writer.add_scalar("objective/entropy", accelerator.gather(mean_entropy).mean().item(), update)
            writer.add_scalar("objective/non_score_reward", accelerator.gather(mean_non_score_reward).mean().item(), update)
            writer.add_scalar(
                "objective/score_total", accelerator.gather(mean_non_score_reward + scores.mean()).mean().item(), update
            )
            writer.add_scalar("objective/scores", accelerator.gather(scores.mean()).mean().item(), update)
            writer.add_scalar("objective/reference_scores", accelerator.gather(reference_scores.mean()).mean().item(), update)
            writer.add_scalar("ppo/loss/policy", accelerator.gather(pg_loss).mean().item(), update)
            writer.add_scalar("ppo/loss/value", accelerator.gather(vf_loss).mean().item(), update)
            writer.add_scalar("ppo/loss/total", accelerator.gather(loss).mean().item(), update)
            writer.add_scalar("ppo/policy/entropy", accelerator.gather(entropy.mean()).mean().item(), update)
            writer.add_scalar("ppo/policy/approxkl", accelerator.gather(approxkl).mean().item(), update)
            writer.add_scalar("ppo/policy/clipfrac", accelerator.gather(pg_clipfrac).mean().item(), update)
            writer.add_scalar("ppo/policy/approxkl_avg", accelerator.gather(approxkls_stats).mean().item(), update)
            writer.add_scalar("ppo/policy/clipfrac_avg", accelerator.gather(clipfracs_stats).mean().item(), update)
            writer.add_scalar("ppo/loss/policy_avg", accelerator.gather(pg_losses_stats).mean().item(), update)
            writer.add_scalar("ppo/loss/value_avg", accelerator.gather(vf_losses_stats).mean().item(), update)
            writer.add_scalar("ppo/val/clipfrac_avg", accelerator.gather(vf_clipfrac_stats).mean().item(), update)
            writer.add_scalar("ppo/policy/entropy_avg", accelerator.gather(entropies_stats).mean().item(), update)
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
            if args.rewards.use_adaptive_kl:
                kl_ctl.update(mean_kl.item(), args.ppo.batch_size)
            del kl, mean_kl, mean_entropy, mean_non_score_reward, scores

    # save model
    if accelerator.is_main_process and args.save_path:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        torch.save(policy.state_dict(), args.save_path)

        if args.upload_model:
            repo_name = f"{args.exp_name}__{args.rewards.label_dataset}__seed{args.seed}__{int(time.time())}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            policy.save_pretrained(repo_id, safe_serialization=True, push_to_hub=True)
            tokenizer.save_pretrained(repo_id, push_to_hub=True)

# if __name__ == "__main__":
#     args = tyro.cli(Args)   
#     train(args)
