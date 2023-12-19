import collections
import functools
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
from tqdm import tqdm
from accelerate import Accelerator
from datasets import load_dataset
from rich.console import Console
from rich.pretty import pprint
from rich.table import Table
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    get_scheduler,
)


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
    truncate_token: int = 50256  # EOS token
    truncate_after: int = 16
    penalty_reward_value: int = -1

    # LM params
    temperature: float = 0.01


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
    run_name: Optional[str] = None
    """a unique name of this run"""
    load_from_cache_file: bool = False
    """Whether to load data from the local cache file in `dataset.map`"""
    push_to_hub: bool = False
    "whether to upload the saved model to huggingface"
    hf_entity: str = ""
    "the user or org name of the model repository from the Hugging Face Hub"
    deepspeed: bool = False
    """Whether to use deepspeed to train the model"""
    print_sample_output_freq: int = 220
    """How often to print sample output"""
    run_eval: bool = False
    """Whether to run evaluation"""

    # optimizer args
    eps: float = 1e-5
    """the epsilon value for the optimizer"""
    lr: float = 6.35e-5
    """the learning rate"""
    optimizer: Literal["adam", "adamw"] = "adamw"
    """Which optimizer to use"""
    scheduler: str = "cosine"
    """Which scheduler to use"""
    warm_up_steps: int = 0
    """Number of warm up steps for the scheduler"""

    gradient_accumulation_steps: int = 16
    """The number of gradient accumulation steps"""
    local_micro_batch_size: Optional[int] = 1
    """The micro batch size per GPU (HF's `per_device_train_batch_size`)"""
    total_episodes: Optional[int] = None
    """The total number of episodes in the dataset"""
    micro_batch_size: Optional[int] = None
    """The micro batch size across devices (HF's `per_device_train_batch_size` * `world_size`)"""
    local_batch_size: Optional[int] = None
    """The batch size per GPU (HF's `per_device_train_batch_size` * `gradient_accumulation_steps`)"""
    batch_size: Optional[int] = None
    """The batch size across devices (HF's `per_device_train_batch_size` * `world_size` * `gradient_accumulation_steps`)"""
    local_eval_batch_size: int = 4
    """per rank eval batch size"""
    world_size: Optional[int] = None
    """The number of processes (GPUs) to use"""
    num_train_epochs: int = 1
    """Number of epochs to train"""
    num_updates: Optional[int] = None
    """The number of updates to train"""
    # other args
    base_model: str = "EleutherAI/pythia-160m"
    """the name of the pretrained model to use"""
    dropout_layer_keys: List[str] = field(default_factory=lambda: ["attn_pdrop", "embd_pdrop", "resid_pdrop", "summary_first_dropout"])
    """Which layers to apply dropout to"""
    output_dir: str = "models/sft_model"
    """Where to save the model"""
    task: TaskHParams = field(default_factory=TaskHParams)


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


def forward(model, query_responses, tokenizer):
    attention_mask = query_responses != tokenizer.pad_token_id
    # position_ids = attention_mask.cumsum(1) - attention_mask.long()  # exclusive cumsum
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        # position_ids=position_ids,
        return_dict=True,
    )


# def train(args: Args):
if __name__ == "__main__":
    args = tyro.cli(Args)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    local_seed = args.seed + accelerator.process_index * 100003  # Prime
    args.world_size = accelerator.num_processes
    args.local_batch_size = args.local_micro_batch_size * args.gradient_accumulation_steps
    args.micro_batch_size = int(args.local_micro_batch_size * args.world_size)
    args.batch_size = int(args.local_batch_size * args.world_size)

    # load dataset
    dataset = load_dataset(args.task.query_dataset, split="train")
    dataset = dataset.shuffle(seed=local_seed)
    dataset = dataset.with_format("torch", columns=["query_token", "reference_response_token"])
    dataloader = DataLoader(dataset, batch_size=args.local_micro_batch_size)
    validation_dataset = load_dataset(args.task.query_dataset, split="validation")
    validation_dataset = validation_dataset.with_format("torch", columns=["query_token", "reference_response_token"])
    validation_dataloader = DataLoader(validation_dataset, batch_size=args.local_eval_batch_size)
    accelerator.print("The number of samples in dataset", len(dataset))
    accelerator.print("The number of samples in validation_dataset", len(validation_dataset))
    args.total_episodes = len(dataset)
    args.num_updates = args.total_episodes // args.local_batch_size

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
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        config=model_config,
        trust_remote_code=True,
    )
    model.generation_config.eos_token_id = None  # disable `pad_token_id` and `eos_token_id` because we just want to
    model.generation_config.pad_token_id = None  # generate tokens without truncation / padding
    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=args.eps)
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, eps=args.eps)
    scheduler = get_scheduler(
        args.scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.warm_up_steps,
        num_training_steps=args.num_updates * args.num_train_epochs,
    )

    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )
    validation_dataloader = accelerator.prepare(validation_dataloader)
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

    accelerator.print("===training model===")
    loss_stats = torch.zeros(args.gradient_accumulation_steps, device=device)
    model.train()
    gradient_accumulation_idx = 0
    global_step = 0
    update = 0
    for epoch in range(args.num_train_epochs):
        accelerator.print(f"epoch: {epoch}")
        for data in dataloader:
            update += 1
            global_step += args.micro_batch_size
            reference_responses = data["reference_response_token"].to(device, non_blocking=True)
            queries = data["query_token"].to(device, non_blocking=True)
            query_responses = torch.cat((queries, reference_responses), dim=1)
            with accelerator.accumulate(model):
                output = forward(model, query_responses, tokenizer)
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
            gradient_accumulation_idx = (gradient_accumulation_idx + 1) % args.gradient_accumulation_steps
            if update > 1 and (update - 1) % args.gradient_accumulation_steps == 0:
                writer.add_scalar("loss", accelerator.gather(loss_stats).mean().item(), update)
                writer.add_scalar("lr", scheduler.get_last_lr()[0], update)
                accelerator.print(f"{loss.item()=}, {scheduler.get_last_lr()=}, {update=}")
            break

    if args.run_eval:
        model.eval()
        rouge_scores = collections.defaultdict(list)
        all_decode_validation_queries = []
        all_decode_validation_query_responses = []
        all_decode_validation_responses = []
        all_decode_validation_reference_responses = []
        all_validation_losses = []
        for validation_idx, validation_data in tqdm(enumerate(validation_dataloader)):
            with torch.no_grad():
                validation_reference_responses = validation_data["reference_response_token"].to(device, non_blocking=True)
                validation_queries = validation_data["query_token"].to(device, non_blocking=True)
                validation_query_reference_responses = torch.cat(
                    (validation_queries, validation_reference_responses), dim=1
                )

                validation_output = forward(model, validation_query_reference_responses, tokenizer)
                validation_labels = validation_query_reference_responses.masked_fill(
                    validation_query_reference_responses == tokenizer.pad_token_id, -1
                )
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
                    accelerator.unwrap_model(model),
                    validation_queries,
                    tokenizer,
                    generation_config,
                )
                decode_validation_queries = tokenizer.batch_decode(accelerator.gather(validation_queries))
                decode_validation_query_responses = tokenizer.batch_decode(accelerator.gather(generated_responses))
                decode_validation_reference_responses = tokenizer.batch_decode(
                    accelerator.gather(validation_reference_responses),
                    skip_special_tokens=True,
                )
                decode_validation_responses = tokenizer.batch_decode(
                    accelerator.gather(generated_responses[:, -args.task.response_length:]),
                    skip_special_tokens=True,
                )
                rouge_score = rouge.compute(
                    predictions=decode_validation_responses, references=decode_validation_reference_responses
                )
                rouge_scores["rouge1"].append(rouge_score["rouge1"])
                rouge_scores["rouge2"].append(rouge_score["rouge2"])
                rouge_scores["rougeL"].append(rouge_score["rougeL"])

                all_decode_validation_queries.extend(decode_validation_queries)
                all_decode_validation_query_responses.extend(decode_validation_query_responses)
                all_decode_validation_responses.extend(decode_validation_responses)
                all_decode_validation_reference_responses.extend(decode_validation_reference_responses)
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

    # save model
    if args.output_dir:
        os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)
        time_tensor = torch.tensor([int(time.time())], device=device)
        time_int = accelerator.gather(time_tensor)[0].item() # avoid different timestamps across processes
        repo_name = f"{args.base_model.replace('/', '_')}__{args.exp_name}__tldr"
        repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name

        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir, repo_id=repo_id)
            if args.push_to_hub:
                tokenizer.push_to_hub(repo_id, revision=f"seed{args.seed}_{str(time_int)}")

        unwrapped: PreTrainedModel = accelerator.unwrap_model(model)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrapped.save_pretrained(
                args.output_dir,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
                state_dict=accelerator.get_state_dict(model),
                safe_serialization=False,
                repo_id=repo_id,
            )
            if args.push_to_hub:
                unwrapped.push_to_hub(repo_id, revision=f"seed{args.seed}_{str(time_int)}", safe_serialization=False)

# if __name__ == "__main__":
#     args = tyro.cli(Args)
#     train(args)
