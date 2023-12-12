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
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)


INVALID_LOGPROB = 1.0


@dataclass
class AdaptiveKLParams:
    target: float = 6.0
    horizon: int = 10000  # in episodes


@dataclass
class RewardHParams:
    use_adaptive_kl: bool = True
    adaptive_kl: Optional[AdaptiveKLParams] = field(default_factory=AdaptiveKLParams)
    trained_model: Optional[str] = ""
    label_dataset: tyro.conf.Suppress[Optional[str]] = None
    dataset_mean: float = 0.
    dataset_std: float = 1.
    kl_coef: float = 0.15


@dataclass
class PpoHParams:
    total_episodes: int = 1000000
    local_batch_size: int = 64
    local_mini_batch_size: tyro.conf.Suppress[int] = None
    batch_size: tyro.conf.Suppress[int] = None
    mini_batch_size: tyro.conf.Suppress[int] = None
    gradient_accumulation_steps: int = 64
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
    query_dataset: str = "vwxyzjn/summarize_from_feedback_tldr_3_filtered_oai_preprocessing_pythia-160m_53"

    query_format_str: Optional[str] = "SUBREDDIT: r/{subreddit}\n\nTITLE: {title}\n\nPOST: {post}\n\nTL;DR:"
    query_truncate_field: Optional[str] = "post"
    query_truncate_text: Optional[str] = "\n"
    query_padding: Optional[str] = None  # defaults to repeated spaces
    query_pad_side: Optional[str] = "left"

    # Response params
    response_length: int = 53

    # Truncate response after the first occurrence of this token at or after index after when sampling.
    truncate_token: Literal["eos"] = "eos"
    truncate_token_id: Optional[int] = None
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
    upload_model: bool = False
    "whether to upload the saved model to huggingface"
    hf_entity: str = ""
    "the user or org name of the model repository from the Hugging Face Hub"

    base_model: str = "EleutherAI/pythia-160m"
    """the name of the pretrained model to use"""
    dropout_layer_keys: List[str] = field(default_factory=lambda: ["attn_pdrop", "embd_pdrop", "resid_pdrop", "summary_first_dropout"])
    """Which layers to apply dropout to"""
    deepspeed: bool = False
    """Whether to use deepspeed to train the model"""
    print_sample_output_freq: int = 10
    """How often to print sample output"""
    save_path: str = "models/ppo_policy"
    """Where to save the model"""
    optimizer: Literal["tf_adam", "adam", "adamw"] = "adamw"
    """Which optimizer to use"""
    sft_model_path: str = ""
    """Where to load the SFT model"""
    task: TaskHParams = field(default_factory=TaskHParams)
    rewards: RewardHParams = field(default_factory=RewardHParams)
    ppo: PpoHParams = field(default_factory=PpoHParams)


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
        # self.scalar_head = layer_init(
        #     nn.Linear(lm_backbone.config.hidden_size, 1),
        #     std=1 / np.sqrt(lm_backbone.config.hidden_size + 1),
        # )
        self.scalar_head = layer_init(nn.Linear(lm_backbone.config.hidden_size, 1), std=0)
        # self.reward_gain = torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.reward_bias = torch.nn.Parameter(torch.tensor(0.0), requires_grad=False)

    def forward(self, **kwargs):
        output = self.lm_backbone(**kwargs)
        reward = self.scalar_head(output.hidden_states[-1]) - self.reward_bias
        return reward


# taken from https://github.com/OpenLMLab/MOSS-RLHF/blob/40b91eb2f2b71b16919addede0341d2bef70825d/ppo/ppo_trainer.py#L29
# we did this we can do a single `model = accelerator.prepare(model)`
class PolicyAndValueWrapper(nn.Module):
    def __init__(self, policy, critic) -> None:
        super().__init__()
        self.policy = policy
        self.critic = critic

    def forward(self, **kwargs):
        return self.policy(**kwargs), self.critic(**kwargs)


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
    input_ids = torch.masked_fill(queries, ~attention_mask, 0)
    output = lm_backbone.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        # position_ids=attention_mask.cumsum(1) - attention_mask.long(), # generation collapsed if this was turned on. TODO: why does generation collapse with this?
        generation_config=generation_config,
        return_dict_in_generate=True,
    )
    return torch.cat((queries, output.sequences[:, context_length:]), dim=1)


def first_true_indices(bools, dtype=torch.long):
    """
    Takes an N-dimensional bool tensor and returns an (N-1)-dimensional tensor of integers giving
    the position of the first True in each "row".

    Returns the length of the rows (bools.size(-1)) if no element is True in a given row.
    """
    row_len = bools.size(-1)
    zero_or_index = row_len * (~bools).type(dtype) + torch.arange(row_len, dtype=dtype, device=bools.device)
    return torch.min(zero_or_index, dim=-1).values


def truncate_response(args, tokenizer, responses):
    trunc_idxs = first_true_indices(responses == args.task.truncate_token_id).unsqueeze(-1)
    new_size = [1] * (len(responses.size()) - 1) + [args.task.response_length]
    idxs = torch.arange(args.task.response_length, device=responses.device).view(*new_size)
    postprocessed_responses = torch.masked_fill(responses, idxs > trunc_idxs, tokenizer.pad_token_id)
    return postprocessed_responses


def get_reward(reward_model, query_responses, tokenizer):
    attention_mask = query_responses != tokenizer.pad_token_id
    # position_ids = attention_mask.cumsum(1) - attention_mask.long()  # exclusive cumsum
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    reward_logits = reward_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        # position_ids=position_ids,
        return_dict=True,
        output_hidden_states=True,
    )
    sequence_lengths = first_true_indices(query_responses == tokenizer.pad_token_id) - 1
    # sequence_lengths1 = (
    #     torch.eq(query_responses, tokenizer.pad_token_id).long().argmax(-1) - 1).to(
    #     query_responses.device
    # )
    # print(f"======={sequence_lengths1=} {sequence_lengths=}")
    # https://github.com/huggingface/transformers/blob/dc68a39c8111217683bf49a4912d0c9018bab33d/src/transformers/models/gpt2/modeling_gpt2.py#L1454
    return reward_logits, reward_logits[torch.arange(reward_logits.size(0), device=reward_logits.device), sequence_lengths].squeeze(-1), sequence_lengths


def forward(policy, query_responses, tokenizer):
    attention_mask = query_responses != tokenizer.pad_token_id
    # position_ids = attention_mask.cumsum(1) - attention_mask.long()  # exclusive cumsum
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    return policy(
        input_ids=input_ids,
        attention_mask=attention_mask,
        # position_ids=position_ids,
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
    if args.ppo.whiten_rewards:
        assert (
            args.ppo.local_mini_batch_size >= 8
        ), f"Per-rank minibatch size {args.ppo.local_mini_batch_size} is insufficient for whitening"
    # `per_rank_rollout_batch_size` is our `args.ppo.local_batch_size`
    # `per_rank_minibatch_size` is our `args.ppo.local_mini_batch_size`
    args.ppo.num_updates = args.ppo.total_episodes // args.ppo.batch_size
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        padding_side="right",
        trust_remote_code=True,
    )
    # we use the padding token manually but do not resize the token embedding of the model
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if args.task.truncate_token == "eos":
        args.task.truncate_token_id = tokenizer.eos_token_id

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

    model_config = AutoConfig.from_pretrained(args.base_model)
    configure_dropout(model_config, args.dropout_layer_keys, 0.0)  # disable dropout
    if accelerator.is_main_process:
        pprint(model_config)
    critic = AutoModelForCausalLMWithRewardHead(
        AutoModelForCausalLM.from_pretrained(
            args.base_model,
            config=model_config,
            trust_remote_code=True,
        )
    )
    reward_model = AutoModelForCausalLMWithRewardHead(
        AutoModelForCausalLM.from_pretrained(
            args.base_model,
            config=model_config,
            trust_remote_code=True,
        )
    )
    if args.rewards.trained_model:
        critic.load_state_dict(torch.load(args.rewards.trained_model, map_location=device), strict=False)
        critic.reward_bias.data = torch.tensor(args.rewards.dataset_mean)
        reward_model.load_state_dict(torch.load(args.rewards.trained_model, map_location=device), strict=False)
        reward_model.reward_bias.data = torch.tensor(args.rewards.dataset_mean)
        print(f"loaded pretrained reward model from {args.rewards.trained_model}")
    # each class should have a separate pretrained model that do not share weights
    ref_policy = AutoModelForCausalLM.from_pretrained(args.base_model, config=model_config, trust_remote_code=True)
    policy = AutoModelForCausalLM.from_pretrained(args.base_model, config=model_config, trust_remote_code=True)
    policy.gradient_checkpointing_enable()
    accelerator.print(policy)
    critic.lm_backbone.gradient_checkpointing_enable()
    accelerator.print(critic)
    if args.sft_model_path:
        policy.load_state_dict(torch.load(args.sft_model_path, map_location=device))
        ref_policy.load_state_dict(torch.load(args.sft_model_path, map_location=device))
        print(f"loaded pretrained policy from {args.sft_model_path}")
    policy.generation_config.eos_token_id = None  # disable `pad_token_id` and `eos_token_id` because we just want to
    policy.generation_config.pad_token_id = None  # generate tokens without truncation / padding
    model = PolicyAndValueWrapper(policy, critic)
    if args.optimizer == "tf_adam":
        optimizer = AdamTensorFlowStyle(model.parameters(), lr=args.ppo.lr, eps=args.ppo.eps)
    elif args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.ppo.lr, eps=args.ppo.eps)
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.ppo.lr, eps=args.ppo.eps)

    dataset = load_dataset(args.task.query_dataset, split="train")
    validation_dataset = load_dataset(args.task.query_dataset, split="validation")
    dataset = dataset.with_format("torch", columns=["query_token", "reference_response_token"])
    dataset = dataset.shuffle(seed=local_seed)
    dataloader = DataLoader(dataset, batch_size=args.ppo.local_batch_size)
    validation_dataset = validation_dataset.with_format("torch", columns=["query_token", "reference_response_token"])
    validation_dataloader = DataLoader(validation_dataset, batch_size=args.ppo.local_batch_size)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    validation_dataloader = accelerator.prepare(validation_dataloader)
    if args.deepspeed:
        import deepspeed

        deepspeed_states = AcceleratorState().deepspeed_plugin
        # deepspeed_states.deepspeed_config["train_micro_batch_size_per_gpu"] = args.ppo.local_micro_batch_size
        # deepspeed_states.deepspeed_config["checkpoint"] = {"use_node_local_storage": True}

        offload = False
        eval_ds_config = {
            "train_micro_batch_size_per_gpu": deepspeed_states.deepspeed_config["train_micro_batch_size_per_gpu"],
            "bf16": {"enabled": True},
            "prescale_gradients": False,
            "wall_clock_breakdown": False,
        }
        if offload:
            eval_ds_config["zero_optimization"] = {
                "stage": 3,
                "stage3_param_persistence_threshold": 1e4,
                "offload_param": {
                    "device": "cpu"
                }
            }
        accelerator.print(f"{eval_ds_config=}")
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

    sample_validation_inds = np.arange(args.ppo.batch_size)
    local_sample_validation_inds = sample_validation_inds[accelerator.process_index :: accelerator.num_processes]
    sample_validation = validation_dataset[local_sample_validation_inds]
    sample_validation_queries = torch.Tensor(sample_validation["query_token"]).to(device)
    with torch.no_grad():
        # sample_validation_queries = shift_pad_id_left(sample_validation_queries, tokenizer.pad_token_id)
        sample_validation_reference_response = torch.Tensor(sample_validation["reference_response_token"]).to(device)
        sample_validation_query_reference_responses = torch.cat(
            (sample_validation_queries, sample_validation_reference_response), dim=1
        )
        # sample_validation_query_reference_responses = shift_pad_id_left(
        #     sample_validation_query_reference_responses, tokenizer.pad_token_id
        # )
        _, sample_validation_reference_scores, _ = get_reward(
            reward_model, sample_validation_query_reference_responses, tokenizer
        )

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
    # use the same `0.01` temperature for validation response generation https://github.com/openai/summarize-from-feedback/blob/700967448d10004279f138666442bf1497d0e705/exps/sample.py#L27
    validation_generation_config= GenerationConfig(
        max_new_tokens=args.task.response_length,
        min_new_tokens=args.task.response_length,
        temperature=(0.01 + 1e-7),
        top_k=0.0,
        top_p=1.0,
        do_sample=True,
    )

    print("===training policy===")
    global_step = 0
    stats_shape = (args.ppo.noptepochs, args.ppo.nminibatches, args.ppo.gradient_accumulation_steps)
    approxkl_stats = torch.zeros(stats_shape, device=device)
    pg_clipfrac_stats = torch.zeros(stats_shape, device=device)
    pg_loss_stats = torch.zeros(stats_shape, device=device)
    vf_loss_stats = torch.zeros(stats_shape, device=device)
    vf_clipfrac_stats = torch.zeros(stats_shape, device=device)
    entropy_stats = torch.zeros(stats_shape, device=device)
    ratio_stats = torch.zeros(stats_shape, device=device)
    model.train()
    for update in range(1, args.ppo.num_updates + 1):
        global_step += 1 * args.ppo.batch_size
        frac = 1.0 - (update - 1.0) / args.ppo.num_updates
        lrnow = frac * args.ppo.lr
        optimizer.param_groups[0]["lr"] = lrnow
        data = next(iter_dataloader)
        with torch.no_grad():
            queries = data["query_token"].to(device)
            query_responses = generate(
                accelerator.unwrap_model(model).policy,
                queries,
                tokenizer,
                generation_config,
            )
            context_length = queries.shape[1]
            responses = query_responses[:, context_length:]

            # validation
            sample_validation_query_responses = generate(
                accelerator.unwrap_model(model).policy,
                sample_validation_queries,
                tokenizer,
                validation_generation_config,
            )
            sample_validation_responses = sample_validation_query_responses[:, context_length:]
            postprocessed_sample_validation_responses = truncate_response(args, tokenizer, sample_validation_responses)
            postprocessed_sample_validation_query_responses = torch.cat(
                (sample_validation_queries, postprocessed_sample_validation_responses), 1
            )
            torch.cuda.empty_cache()

            # TODO: do I do this with query response or post-processed query response?
            output = forward(accelerator.unwrap_model(model).policy, query_responses, tokenizer)
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
            postprocessed_responses = truncate_response(args, tokenizer, responses)
            torch.cuda.empty_cache()

            # 2. run reward model on the truncated responses
            postprocessed_query_responses = torch.cat((queries, postprocessed_responses), 1)
            # sequence_lengths = first_true_indices(postprocessed_responses == tokenizer.pad_token_id) - 1
            # actual_start = torch.arange(postprocessed_responses.size(0), device=postprocessed_responses.device)
            # actual_end = sequence_lengths
            # padding_mask = postprocessed_responses == tokenizer.pad_token_id

            full_values, _, _ = get_reward(accelerator.unwrap_model(model).critic, query_responses, tokenizer)
            values = full_values[:, context_length - 1 : -1].squeeze(-1)
            # values_mask = postprocessed_responses != args.task.truncate_token_id
            # values = torch.masked_fill(values, values_mask, 0)
            # values = torch.masked_fill(values, padding_mask, 0)

            # logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
            # ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)
            _, scores, _ = get_reward(reward_model, postprocessed_query_responses, tokenizer)

            _, validation_score, _ = get_reward(reward_model, postprocessed_sample_validation_query_responses, tokenizer)

            # 3. filter response. Ensure that the sample contains truncate_token_id
            # responses not passing that filter will receive a low (fixed) score
            # only query humans on responses that pass that filter
            contain_pad_token = torch.any(postprocessed_responses == tokenizer.pad_token_id, dim=-1)
            scores = torch.where(contain_pad_token, scores, torch.full_like(scores, args.task.penalty_reward_value))
            
            # TODO: do we need to deal with penalty values?
            # penalty_values = torch.full_like(values, 0)
            # penalty_values[:,-1] += args.task.penalty_reward_value
            # values = torch.where(contain_pad_token, values, penalty_values)
            accelerator.print(f"{scores=}, {(contain_pad_token.sum() / len(contain_pad_token))=}")
            # torch.cuda.empty_cache()

            # 4. compute rewards
            kl = logprobs - ref_logprobs
            # kl = torch.masked_fill(kl, padding_mask, 0)
            non_score_reward = -kl_ctl.value * kl
            rewards = non_score_reward.clone()
            rewards[:, -1] += scores

            # 5. whiten rewards
            if args.ppo.whiten_rewards:
                rewards = whiten(rewards, shift_mean=False)

            if args.print_sample_output_freq > 0 and (update - 1) % args.print_sample_output_freq == 0:
                try:
                    all_decode_validation_queries = tokenizer.batch_decode(sample_validation_queries, skip_special_tokens=True)
                    all_sample_validation_responses = tokenizer.batch_decode(sample_validation_responses)
                    all_sample_validation_query_responses_postprocessed = tokenizer.batch_decode(
                        postprocessed_sample_validation_query_responses, skip_special_tokens=True
                    )
                    all_sample_validation_postprocessed_responses = [
                        x[len(y) :]
                        for x, y in zip(all_sample_validation_query_responses_postprocessed, all_decode_validation_queries)
                    ]
                    all_sample_validation_reference_responses = tokenizer.batch_decode(sample_validation_reference_response)
                    all_sample_validation_df = pd.DataFrame(
                        {
                            "query": all_decode_validation_queries,
                            "response": all_sample_validation_responses,
                            "postprocessed_response": all_sample_validation_postprocessed_responses,
                            "reference_responses": all_sample_validation_reference_responses,
                            "scores": validation_score.float().cpu().numpy(),
                            "reference_scores": sample_validation_reference_scores.float().cpu().numpy(),
                        }
                    )
                    if accelerator.is_main_process:
                        all_sample_validation_df.to_json(f"runs/{run_name}/table.json")
                        if args.track:
                            wandb.log({"samples/query_responses": wandb.Table(dataframe=all_sample_validation_df)}, step=update)
                    print_rich_table("stuff", all_sample_validation_df[:4], console)

                except Exception as e:
                    print(e)
                del (
                    all_decode_validation_queries,
                    all_sample_validation_responses,
                    all_sample_validation_reference_responses,
                    all_sample_validation_df,
                )
            # del postprocessed_query_responses
            # torch.cuda.empty_cache()

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
            writer.add_histogram("rewards", rewards[0].float(), global_step)
            writer.add_histogram("advantages", advantages[0].float(), global_step)
            accelerator.print("rewards====", rewards[0])
            accelerator.print("advantages====", advantages[0])
            # raise
            # pprint({
            #     "rewards": rewards,
            #     "returns": returns,
            #     "advantages": advantages,
            # })
            # breakpoint()
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

                        output, vpred_temp = forward(model, mb_query_responses, tokenizer)
                        logits = output.logits[:, context_length - 1 : -1]
                        logits /= (args.task.temperature + 1e-7)
                        new_all_logprobs = F.log_softmax(logits, dim=-1)
                        new_logprobs = torch.gather(new_all_logprobs, 2, mb_responses.unsqueeze(-1)).squeeze(-1)
                        # new_logprobs = torch.masked_fill(new_logprobs, padding_mask[micro_batch_inds], INVALID_LOGPROB)
                        vpred = vpred_temp[:, context_length - 1 : -1].squeeze(-1)
                        # vpred = torch.masked_fill(vpred, padding_mask[micro_batch_inds], 0)
                        # vpred = torch.masked_fill(vpred, values_mask[micro_batch_inds], 0)
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
                        # if ppo_epoch_idx == 0 and micro_batch_start == 0:
                        #     torch.testing.assert_close(ratio, torch.zeros_like(ratio) + 1, atol=1e-4, rtol=1e-4)
                        # if ppo_epoch_idx == 0: 
                        #     pprint({
                        #         # "responses": responses,
                        #         # "values": values,
                        #         "rewards": rewards,
                        #         # "scores": scores,
                        #         "advantages": advantages,
                        #         # "ratio": ratio,
                        #         # "pg_losses": pg_losses,
                        #         # "approxkl": approxkl,
                        #         # "pg_loss": pg_loss,
                        #         # "pg_clipfrac": pg_clipfrac,
                        #         # "ratio": ratio.mean(),
                        #         # "vf_loss": vf_loss,
                        #         # "vf_clipfrac": vf_clipfrac,
                        #         # "entropy": masked_mean(entropy, ~padding_mask[micro_batch_inds]),
                        #     })
                        #     breakpoint()
                        with torch.no_grad():
                            approxkl_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = approxkl
                            pg_clipfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = pg_clipfrac
                            pg_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = pg_loss
                            vf_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = vf_loss
                            vf_clipfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = vf_clipfrac
                            entropy_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = entropy.mean()
                            ratio_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = ratio.mean()
                    gradient_accumulation_idx += 1
                minibatch_idx += 1
                if accelerator.is_main_process:
                    console.print(
                        f"ppo_epoch_idx",
                        ppo_epoch_idx,
                        "approxkl",
                        approxkl_stats[:ppo_epoch_idx+1].mean().item(),
                        "pg_loss",
                        pg_loss_stats[:ppo_epoch_idx+1].mean().item(),
                        "pg_clipfrac",
                        pg_clipfrac_stats[:ppo_epoch_idx+1].mean().item(),
                        "ratio",
                        ratio_stats[:ppo_epoch_idx+1].mean().item(),
                    )
        # raise
        # breakpoint()
        with torch.no_grad():
            if not args.deepspeed:  # for some reason there is a OOM with the `writer.add_histogram`
                writer.add_histogram("ppo/val/ratio_hist", ratio, update)
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
            writer.add_scalar("objective/validation_score", accelerator.gather(validation_score.mean()).mean().item(), update)
            writer.add_scalar("ppo/loss/policy", accelerator.gather(pg_loss).mean().item(), update)
            writer.add_scalar("ppo/loss/value", accelerator.gather(vf_loss).mean().item(), update)
            writer.add_scalar("ppo/loss/total", accelerator.gather(loss).mean().item(), update)
            writer.add_scalar("ppo/policy/entropy", accelerator.gather(entropy.mean()).mean().item(), update)
            writer.add_scalar("ppo/policy/approxkl", accelerator.gather(approxkl).mean().item(), update)
            writer.add_scalar("ppo/policy/clipfrac", accelerator.gather(pg_clipfrac).mean().item(), update)
            writer.add_scalar("ppo/policy/approxkl_avg", accelerator.gather(approxkl_stats).mean().item(), update)
            writer.add_scalar("ppo/policy/clipfrac_avg", accelerator.gather(pg_clipfrac_stats).mean().item(), update)
            writer.add_scalar("ppo/loss/policy_avg", accelerator.gather(pg_loss_stats).mean().item(), update)
            writer.add_scalar("ppo/loss/value_avg", accelerator.gather(vf_loss_stats).mean().item(), update)
            writer.add_scalar("ppo/val/clipfrac_avg", accelerator.gather(vf_clipfrac_stats).mean().item(), update)
            writer.add_scalar("ppo/policy/entropy_avg", accelerator.gather(entropy_stats).mean().item(), update)
            writer.add_scalar("ppo/returns/mean", accelerator.gather(return_mean).mean().item(), update)
            writer.add_scalar("ppo/returns/var", accelerator.gather(return_var).mean().item(), update)
            writer.add_scalar("ppo/val/vpred", accelerator.gather(vpred.mean()).mean().item(), update)
            writer.add_scalar("ppo/val/error", accelerator.gather(vf_losses1.mean()).mean().item(), update)
            writer.add_scalar("ppo/val/clipfrac", accelerator.gather(vf_clipfrac).mean().item(), update)
            writer.add_scalar("ppo/val/mean", accelerator.gather(value_mean).mean().item(), update)
            writer.add_scalar("ppo/val/var", accelerator.gather(value_var).mean().item(), update)
            writer.add_scalar("ppo/val/ratio", accelerator.gather(ratio_stats).mean().item(), update)
            writer.add_scalar("ppo/val/ratio_var", accelerator.gather(ratio_stats).var().item(), update)
            writer.add_scalar("ppo/val/advantage", accelerator.gather(advantages.mean()).mean().item(), update)
            writer.add_scalar("ppo/val/advantage_var", accelerator.gather(advantages.mean()).var().item(), update)
            writer.add_scalar("ppo/val/num_eos_tokens", (responses == tokenizer.eos_token_id).sum().item(), update)
            writer.add_scalar("ppo/lr", lrnow, update)
            writer.add_scalar("ppo/episode", global_step, update)
            if args.rewards.use_adaptive_kl:
                kl_ctl.update(mean_kl.item(), args.ppo.batch_size)
            del kl, mean_kl, mean_entropy, mean_non_score_reward, scores

    # save model
    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        accelerator.save_model(policy, args.save_path, max_shard_size="1000GB")

        if args.upload_model and accelerator.is_main_process:
            repo_name = f"{args.exp_name}__{args.rewards.label_dataset}__seed{args.seed}__{int(time.time())}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            policy.save_pretrained(repo_id, safe_serialization=True, push_to_hub=True)
            tokenizer.save_pretrained(repo_id, push_to_hub=True)

# if __name__ == "__main__":
#     args = tyro.cli(Args)
#     train(args)
