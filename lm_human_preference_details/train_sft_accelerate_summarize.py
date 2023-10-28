import collections
import os
import random
import time
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace
from typing import List, Literal, Optional

import evaluate
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
from torch.nn import functional as F
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
    get_scheduler,
)


@dataclass
class SFTHParams:
    gradient_accumulation_steps: int = 16
    local_micro_batch_size: int = 1
    noptepochs: int = 1
    lr: float = 6.35e-5
    eps: float = 1e-5
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
    query_dataset: str = "vwxyzjn/summarize_from_feedback_tldr_3_filtered_oai_preprocessing"

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
    load_from_cache_file: bool = False
    """Whether to load data from the local cache file in `dataset.map`"""
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
    save_path: str = "models/sft_policy"
    """Where to save the model"""
    optimizer: Literal["tf_adam", "adam", "adamw"] = "adamw"
    """Which optimizer to use"""
    scheduler: str = "cosine"
    """Which scheduler to use"""
    warm_up_steps: int = 0
    """Number of warm up steps for the scheduler"""
    task: TaskHParams = field(default_factory=TaskHParams)
    sft: SFTHParams = field(default_factory=SFTHParams)


# taken from https://github.com/microsoft/DeepSpeedExamples/blob/737c6740bec38b77a24a59135b6481a53d566b38/applications/DeepSpeed-Chat/training/utils/model/model_utils.py#L20C1-L26C52
def configure_dropout(model_config, dropout):
    if dropout is not None:
        for key in ("dropout", "attention_dropout", "hidden_dropout", "activation_dropout"):
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


def shift_pad_id_left(data, pad_id):
    # Step 1: Create a boolean mask
    mask = (data == pad_id).long()
    # Step 3: Use argsort on the inverted boolean mask to get sorted indices
    sorted_indices = torch.argsort(~mask, axis=1)
    # Step 4: Use advanced indexing to rearrange the elements
    rows_range = torch.arange(data.shape[0], device=data.device)
    shifted_data = data[rows_range[:, None], sorted_indices]
    return shifted_data


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


# def train(args: Args):
if __name__ == "__main__":
    args = tyro.cli(Args)
    accelerator = Accelerator(gradient_accumulation_steps=args.sft.gradient_accumulation_steps)
    args.sft.world_size = accelerator.num_processes
    args.sft.local_batch_size = args.sft.local_micro_batch_size * args.sft.gradient_accumulation_steps
    args.sft.batch_size = int(args.sft.local_batch_size * args.sft.world_size)
    dataset = load_dataset(args.task.query_dataset, split="train")
    validation_dataset = load_dataset(args.task.query_dataset, split="validation")
    accelerator.print("The number of samples in dataset", len(dataset))
    accelerator.print("The number of samples in validation_dataset", len(validation_dataset))
    args.sft.total_episodes = len(dataset)
    args.sft.num_updates = args.sft.total_episodes // args.sft.batch_size

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
    configure_dropout(model_config, 0.0)  # disable dropout
    policy = AutoModelForCausalLM.from_pretrained(args.base_model, config=model_config, trust_remote_code=True)
    policy.generation_config.eos_token_id = None  # disable `pad_token_id` and `eos_token_id` because we just want to
    policy.generation_config.pad_token_id = None  # generate tokens without truncation / padding
    # IMPORTANT: Layer norm produces weird gradients, which affects Adam optimizer to impact all the parameters systematically
    # see https://github.com/pytorch/pytorch/issues/104857 for more details
    if args.optimizer == "tf_adam":
        optimizer = AdamTensorFlowStyle(policy.parameters(), lr=args.sft.lr, eps=args.sft.eps)
    elif args.optimizer == "adam":
        optimizer = optim.Adam(policy.parameters(), lr=args.sft.lr, eps=args.sft.eps)
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW(policy.parameters(), lr=args.sft.lr, eps=args.sft.eps)
    scheduler = get_scheduler(
        args.scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.warm_up_steps,
        num_training_steps=args.sft.num_updates // args.sft.gradient_accumulation_steps,
    )

    dataset = dataset.with_format("torch", columns=["query_token", "reference_response_token"])
    dataset = dataset.shuffle(seed=local_seed)
    dataloader = DataLoader(dataset, batch_size=args.sft.local_micro_batch_size)
    validation_dataset = validation_dataset.with_format("torch", columns=["query_token", "reference_response_token"])
    validation_dataloader = DataLoader(validation_dataset, batch_size=args.sft.local_micro_batch_size)
    policy, optimizer, dataloader, validation_dataloader, scheduler = accelerator.prepare(
        policy, optimizer, dataloader, validation_dataloader, scheduler
    )
    iter_dataloader = iter(dataloader)
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
    rouge = evaluate.load("rouge")

    print("===training policy===")
    global_step = 0
    loss_stats = torch.zeros(args.sft.gradient_accumulation_steps, device=device)
    gradient_accumulation_idx = 0
    policy.train()
    for update in range(1, args.sft.num_updates + 1):
        global_step += args.sft.batch_size
        accelerator.print(f"update {update}, global_step {global_step}")
        data = next(iter_dataloader)
        reference_responses = data["reference_response_token"].to(device, non_blocking=True)
        queries = data["query_token"].to(device, non_blocking=True)
        query_responses = torch.cat((queries, reference_responses), dim=1)
        query_responses = shift_pad_id_left(query_responses, tokenizer.pad_token_id)
        with accelerator.accumulate(policy):
            output = forward(policy, query_responses, tokenizer)
            # mask out gradient effects on response padding tokens
            labels = query_responses.masked_fill(query_responses == tokenizer.pad_token_id, -1)
            lm_logits = output.logits
            # hand-rolled transformer loss: Shift so that tokens < n predict n
            # but unlike `transformers` we mask the padding tokens via `ignore_index=-1`
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-1)
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        loss_stats[gradient_accumulation_idx] = loss
        gradient_accumulation_idx = (gradient_accumulation_idx + 1) % args.sft.gradient_accumulation_steps
        if update > 1 and (update - 1) % args.sft.gradient_accumulation_steps == 0:
            writer.add_scalar("loss", accelerator.gather(loss_stats).mean().item(), update)
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], update)
        if update == 1 or update == args.sft.num_updates - 1:
            policy.eval()
            rouge_scores = collections.defaultdict(list)
            all_decode_validation_queries = []
            all_decode_validation_query_responses = []
            all_decode_validation_responses = []
            all_decode_validation_reference_responses = []
            all_validation_losses = []
            for validation_idx, validation_data in enumerate(validation_dataloader):
                with torch.no_grad():
                    validation_reference_responses = validation_data["reference_response_token"].to(device, non_blocking=True)
                    validation_queries = validation_data["query_token"].to(device, non_blocking=True)
                    validation_queries = shift_pad_id_left(validation_queries, tokenizer.pad_token_id)
                    validation_query_reference_responses = torch.cat(
                        (validation_queries, validation_reference_responses), dim=1
                    )

                    validation_output = forward(policy, validation_query_reference_responses, tokenizer)
                    validation_labels = validation_query_reference_responses.masked_fill(
                        validation_query_reference_responses == tokenizer.pad_token_id, -1
                    )
                    if args.sft.lm_loss_on_response_only:
                        validation_labels[:, : queries.shape[1]] = -1
                    validation_lm_logits = validation_output.logits
                    # hand-rolled transformer loss: Shift so that tokens < n predict n
                    # but unlike `transformers` we mask the padding tokens via `ignore_index=-1`
                    validation_shift_logits = validation_lm_logits[..., :-1, :].contiguous()
                    validation_shift_labels = validation_labels[..., 1:].contiguous()
                    validation_loss = F.cross_entropy(
                        validation_shift_logits.view(-1, validation_shift_logits.size(-1)),
                        validation_shift_labels.view(-1),
                        ignore_index=-1,
                    )
                    validation_loss = accelerator.gather(validation_loss)
                    all_validation_losses.append(validation_loss)

                    generated_responses = generate(
                        accelerator.unwrap_model(policy), validation_queries, tokenizer, generation_config
                    )
                    decode_validation_queries = tokenizer.batch_decode(accelerator.gather(validation_queries))
                    decode_validation_query_responses = tokenizer.batch_decode(accelerator.gather(generated_responses))
                    decode_validation_reference_responses = tokenizer.batch_decode(
                        accelerator.gather(validation_reference_responses)
                    )
                    decode_validation_responses = [
                        x[len(y) :] for x, y in zip(decode_validation_query_responses, decode_validation_queries)
                    ]
                    rouge_score = rouge.compute(
                        predictions=decode_validation_responses, references=decode_validation_reference_responses
                    )
                    rouge_scores["rouge1"].append(rouge_score["rouge1"])
                    rouge_scores["rouge2"].append(rouge_score["rouge2"])
                    rouge_scores["rougeL"].append(rouge_score["rougeL"])

                    all_decode_validation_queries.extend(decode_validation_queries)
                    accelerator.print(
                        "len(all_decode_validation_queries)", len(all_decode_validation_queries), decode_validation_responses
                    )
                    all_decode_validation_query_responses.extend(decode_validation_query_responses)
                    all_decode_validation_responses.extend(decode_validation_responses)
                    all_decode_validation_reference_responses.extend(decode_validation_reference_responses)
                if validation_idx == 10:
                    break

            try:
                all_df = pd.DataFrame(
                    {
                        "query": all_decode_validation_queries,
                        "response": all_decode_validation_responses,
                        "reference": all_decode_validation_reference_responses,
                    }
                )
                accelerator.print(all_df)
                if accelerator.is_main_process and args.track:
                    wandb.log({"samples/query_responses": wandb.Table(dataframe=all_df)}, step=update)
                    print_rich_table(f"Sample Output at Step {update}", all_df[:4], console)
            except Exception as e:
                print(e)

            for k, v in rouge_scores.items():
                rouge_metric = torch.tensor(v, device=device)
                rouge_metric = accelerator.gather(rouge_metric)
                writer.add_scalar(f"rouge/{k}", rouge_metric.mean().item(), update)
                accelerator.print(f"rouge/{k}: {rouge_metric.mean().item()} {rouge_metric.shape} {rouge_metric}")
            writer.add_scalar("validation_loss", torch.stack(all_validation_losses).mean().item(), update)
            policy.train()

    # save model
    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        accelerator.save_model(policy, args.save_path, max_shard_size="1000GB")

        if args.upload_model and accelerator.is_main_process:
            repo_name = f"{args.exp_name}__tldr__seed{args.seed}__{int(time.time())}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            policy.save_pretrained(repo_id, safe_serialization=True, push_to_hub=True)
            tokenizer.save_pretrained(repo_id, push_to_hub=True)

# if __name__ == "__main__":
#     args = tyro.cli(Args)
#     train(args)