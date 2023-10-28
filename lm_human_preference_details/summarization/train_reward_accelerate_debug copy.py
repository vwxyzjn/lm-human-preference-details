import os
import random
import time
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, broadcast
from datasets import load_dataset
from rich.console import Console
from rich.pretty import pprint
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from lm_human_preference_details.datamod import DATASET


@dataclass
class LabelHParams:
    type: str = None
    num_train: int = 4992
    num_labels: int = 4
    source: str = None


@dataclass
class TaskHParams:
    # Query params
    query_length: int = 64
    query_dataset: str = "books"
    query_prefix: str = ""
    query_suffix: str = ""
    start_text: Optional[str] = None
    end_text: Optional[str] = None

    # Response params
    response_length: int = 24

    # LM params
    temperature: float = 0.7


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
    label_dataset: str = "sentiment/offline_5k.json"
    """the name of the dataset to use for labels in `https://huggingface.co/datasets/vwxyzjn/lm-human-preferences`"""
    local_batch_size: int = 4
    """per rank batch size"""
    lr: float = 0.00005
    """the learning rate"""
    eps: float = 1e-5
    """the epsilon for AdamW"""
    local_rollout_batch_size: int = 512
    """per rank rollot batch size"""
    world_size: tyro.conf.Suppress[int] = None
    """the number of processes to use"""
    batch_size: tyro.conf.Suppress[int] = None
    """the batch size across all ranks"""
    normalize_samples: int = 256
    """Samples used to estimate reward mean and std"""
    debug_normalize: int = 0
    """Samples used to check that normalization worked"""
    normalize_before: bool = True
    """Whether, before training, to normalize the rewards on the policy to the scales on the training buffer. (For comparisons, just use mean 0, var 1.)"""
    normalize_after: bool = True
    """Whether, after training, to normalize the rewards on the ref policy to mean 0, var 1 (so the KL coefficient always has the same meaning)."""
    print_sample_output_freq: int = 10
    """How often to print sample output"""
    save_path: str = "models/reward.pt"
    """Where to save the model"""
    task: TaskHParams = field(default_factory=TaskHParams)
    labels: LabelHParams = field(default_factory=LabelHParams)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.normal_(layer.weight, std=std)
    torch.nn.init.constant_(layer.bias, val=bias_const)
    return layer


OPENAI_PAD_TOKEN_ID = 50259


class ScalarHead(nn.Module):
    def __init__(self, config, scale=None, **kwargs):
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
        if scale is None:
            scale = 1 / np.sqrt(hidden_size + 1)
        self.summary = layer_init(nn.Linear(hidden_size, 1), std=scale)
        self.flatten = nn.Flatten()

    def forward(self, hidden_states):
        output = self.dropout(hidden_states)
        output = self.summary(output)
        return output


class AutoModelForCausalLMWithScalarHead(nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.scalar_head = ScalarHead(self.pretrained_model.config, scale=0.0)

    def forward(self, **kwargs):
        output = self.pretrained_model(**kwargs)
        return output, self.scalar_head(output.hidden_states[-1])


class AutoModelForCausalLMWithRewardHead(nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.scalar_head = ScalarHead(self.pretrained_model.config)
        self.reward_gain = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.reward_bias = torch.nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def forward(self, **kwargs):
        output = self.pretrained_model(**kwargs)
        reward = self.scalar_head(output.hidden_states[-1])
        reward = self.reward_gain * reward + self.reward_bias
        # but we only care about the reward of the last token
        reward = reward[:, -1]
        return output, reward


# a pytorch dataset
class MyDataset(IterableDataset):
    def __init__(
        self, generator, tokenizer, query_length, start_text=None, end_text=None, query_prefix="", query_suffix="", seed=None
    ):
        self.generator = generator
        self.tokenizer = tokenizer
        self.query_length = query_length
        self.start_text = start_text
        self.end_text = end_text
        self.seed = seed
        token_to_index = tokenizer.get_vocab()
        self.start_token = token_to_index[start_text] if self.start_text else None
        self.end_token = token_to_index[end_text] if self.end_text else None
        self.query_prefix = query_prefix
        self.query_suffix = query_suffix
        self.query_prefix_tokens = torch.LongTensor(tokenizer.encode(query_prefix))
        self.query_suffix_tokens = torch.LongTensor(tokenizer.encode(query_suffix))

    def __iter__(self):
        for text in self.generator("train", self.seed, shuffle=True):
            tokens = self.tokenizer.encode(text)
            if self.start_token is not None:
                try:
                    first_index = tokens.index(self.start_token) + 1
                    if first_index < len(tokens):
                        tokens = tokens[first_index:]
                except:
                    continue
            tokens = tokens[: self.query_length]
            if self.end_token is not None:
                try:
                    last_index = len(tokens) - tokens[::-1].index(self.end_token)
                    tokens = tokens[:last_index]
                except:
                    continue
            output = self.tokenizer.pad(
                {"input_ids": tokens},
                padding="max_length",
                max_length=self.query_length,
                return_tensors="pt",
            )
            output["input_ids"] = torch.cat((self.query_prefix_tokens, output["input_ids"], self.query_suffix_tokens))
            yield output


def left_padding_to_right_padding(query, pad_id):
    # got to convert to right padding, otherwise `transformers` has weird issues
    # even with `position_ids`
    return torch.tensor([[pad_id] * (row == pad_id).sum() + [x for x in row if x != pad_id] for row in query])


def ceil_div(a, b):
    return (a - 1) // b + 1


def generate(pretrained_model, queries, tokenizer, generation_config):
    """generate in a way that does not affect padding tokens"""
    context_length = queries.shape[1]
    attention_mask = queries != tokenizer.pad_token_id
    input_ids = queries.clone()
    input_ids[~attention_mask] = 0  # set padding tokens to 0
    output = pretrained_model.generate(
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
    input_ids = query_responses.clone()
    input_ids[~attention_mask] = 0
    return reward_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        return_dict=True,
        output_hidden_states=True,
    )


def normalize(args, accelerator, device, tokenizer, pretrained_model, reward_model, iter_dataloader, generation_config):
    with torch.no_grad():
        # reset reward scales
        reward_model.module.reward_gain.data.fill_(1.0)
        reward_model.module.reward_bias.data.fill_(0.0)

        # sample queries and responses
        n_batches = ceil_div(args.normalize_samples, args.local_rollout_batch_size)
        sample_queries_responses = []
        for _ in range(n_batches):
            data = next(iter_dataloader)
            queries = data["input_ids"].to(device)
            queries = left_padding_to_right_padding(data["input_ids"], tokenizer.pad_token_id).to(device)
            query_responses = generate(pretrained_model, queries, tokenizer, generation_config)
            sample_queries_responses.append(query_responses)

        # compute reward statistics
        rewards = []
        for query_responses in sample_queries_responses:
            rewards.append(get_reward(reward_model, query_responses, tokenizer)[1])
        rewards = torch.cat(rewards)
        rewards = accelerator.gather(rewards)
        mean, std = rewards.mean(), rewards.std()
        print(f"mean: {mean}, std: {std}")

        # reward normalization
        target_mean, target_std = torch.tensor(0.0, device=device), torch.tensor(1.0, device=device)
        gain = target_std / std
        bias = target_mean - gain * mean
        print(f"gain: {gain}, bias: {bias}")
        reward_model.module.reward_gain.data = gain
        reward_model.module.reward_bias.data = bias

        # after normalization statistics
        n_batches = ceil_div(args.normalize_samples, args.local_rollout_batch_size)
        sample_queries_responses = []
        for _ in range(n_batches):
            data = next(iter_dataloader)
            queries = data["input_ids"].to(device)
            queries = left_padding_to_right_padding(data["input_ids"], tokenizer.pad_token_id).to(device)
            query_responses = generate(pretrained_model, queries, tokenizer, generation_config)
            sample_queries_responses.append(query_responses)
        rewards = []
        for query_responses in sample_queries_responses:
            rewards.append(get_reward(reward_model, query_responses, tokenizer)[1])
        rewards = torch.cat(rewards)
        rewards = accelerator.gather(rewards)
        mean, std = rewards.mean(), rewards.std()
        print(f"after mean: {mean}, after std: {std}")


def train(args: Args):
    args.task.query_prefix = args.task.query_prefix.replace("\\n", "\n")
    args.task.query_suffix = args.task.query_suffix.replace("\\n", "\n")
    accelerator = Accelerator(
        kwargs_handlers=[
            DistributedDataParallelKwargs(broadcast_buffers=False)
        ]  # this is needed to avoid https://github.com/pytorch/pytorch/issues/22095#issuecomment-505099500
    )
    args.world_size = accelerator.num_processes
    args.batch_size = int(args.local_batch_size * args.world_size)

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
    args.seed += accelerator.process_index
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        padding_side="right",
    )
    # we use the padding token manually but do not resize the token embedding of the model
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    untrained_model = AutoModelForCausalLMWithRewardHead(AutoModelForCausalLM.from_pretrained(args.base_model)).to(device)
    reward_model = AutoModelForCausalLMWithRewardHead(AutoModelForCausalLM.from_pretrained(args.base_model)).to(device)
    reward_model.pretrained_model.generation_config.eos_token_id = (
        None  # disable `pad_token_id` and `eos_token_id` because we just want to
    )
    reward_model.pretrained_model.generation_config.pad_token_id = None  # generate tokens without truncation / padding
    optimizer = optim.Adam(reward_model.parameters(), lr=args.lr, eps=args.eps)
    dataset = MyDataset(
        DATASET[args.task.query_dataset],
        tokenizer,
        args.task.query_length,
        start_text=args.task.start_text,
        end_text=args.task.end_text,
        query_prefix=args.task.query_prefix,
        query_suffix=args.task.query_suffix,
    )
    dataloader = DataLoader(dataset, batch_size=args.local_rollout_batch_size)
    reward_model, optimizer, dataloader = accelerator.prepare(reward_model, optimizer, dataloader)
    print(reward_model)
    iter_dataloader = iter(dataloader)

    generation_config = GenerationConfig(
        max_new_tokens=args.task.response_length,
        min_new_tokens=args.task.response_length,
        temperature=args.task.temperature,
        top_k=0.0,
        top_p=1.0,
        do_sample=True,
    )

    # `label` has keys `['sample0', 'query', 'best', 'sample3', 'sample1', 'sample2']`
    label = load_dataset(
        "vwxyzjn/lm-human-preferences",
        data_files=[args.label_dataset],
    )["train"]
    print("Num labels found in source:", len(label))
    print("training on", args.labels.num_train, "in batches of", args.local_batch_size)

    print("before====", reward_model.module.reward_gain.data)
    if args.normalize_before:
        normalize(
            args,
            accelerator,
            device,
            tokenizer,
            accelerator.unwrap_model(reward_model).pretrained_model,
            reward_model,
            iter_dataloader,
            generation_config,
        )
    print("after====", reward_model.module.reward_gain.data)

    print("===training reward model===")
    all_inds = np.arange(args.labels.num_train)
    np.random.shuffle(all_inds)
    # ensure that all processes have the same shuffled indices
    all_inds = broadcast(torch.tensor(all_inds, device=device), 0)
    all_inds = all_inds.cpu().numpy()
    global_step = 0
    for start in range(0, args.labels.num_train, args.batch_size):
        global_step += 1
        end = start + args.batch_size
        b_inds_all = all_inds[start:end]
        b_inds = b_inds_all[accelerator.process_index :: accelerator.num_processes]  #  multi-GPU slicing
        lr = (1 - start / args.labels.num_train) * args.lr
        optimizer.param_groups[0]["lr"] = lr
        mb_data = label[b_inds]
        # print("accelerator.process_index", accelerator.process_index, b_inds, b_inds_all)
        mb_query = torch.from_numpy(np.stack(mb_data["query"]))
        print("mb_query.shape", mb_query.shape)
        mb_query = left_padding_to_right_padding(mb_query, tokenizer.pad_token_id).to(device)
        mb_best = torch.from_numpy(np.stack(mb_data["best"])).to(device)
        mb_responses = [torch.from_numpy(np.stack(mb_data[f"sample{i}"])).to(device) for i in range(args.labels.num_labels)]
        # hack: deal with openai's padding token
        # assert (mb_query == tokenizer.pad_token_id).sum() == 0
        mb_query[mb_query == OPENAI_PAD_TOKEN_ID] = tokenizer.pad_token_id
        for item in mb_responses:
            # assert (item == tokenizer.pad_token_id).sum() == 0
            item[item == OPENAI_PAD_TOKEN_ID] = tokenizer.pad_token_id

        predicted_rewards = []
        for i in range(args.labels.num_labels):
            query_responses = torch.cat([mb_query, mb_responses[i]], dim=1)
            reward = get_reward(reward_model, query_responses, tokenizer)[1]
            predicted_rewards.append(reward.squeeze())
        predicted_rewards = torch.stack(
            predicted_rewards, dim=1
        )  # shape (batch_size, num_labels), basically a reward prediction for each label
        accuracy = (predicted_rewards.argmax(1) == mb_best).float().mean()
        loss = torch.nn.functional.cross_entropy(predicted_rewards, mb_best)
        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()
        writer.add_scalar("train/loss", accelerator.gather(loss).mean().item(), global_step)
        writer.add_scalar("train/accuracy", accelerator.gather(accuracy).mean().item(), global_step)
        writer.add_scalar("train/lr", lr, global_step)

        if args.print_sample_output_freq > 0 and global_step % args.print_sample_output_freq == 0:
            with torch.no_grad():
                data = next(iter_dataloader)
                queries = data["input_ids"].to(device)
                context_length = queries.shape[1]
                queries = left_padding_to_right_padding(data["input_ids"], tokenizer.pad_token_id).to(device)
                query_responses = generate(
                    accelerator.unwrap_model(reward_model).pretrained_model, queries, tokenizer, generation_config
                )
                responses = query_responses[:, context_length:]

                output, reward = get_reward(reward_model, query_responses, tokenizer)
                logits = output.logits[:, context_length - 1 : -1]
                logits /= args.task.temperature
                all_logprobs = F.log_softmax(logits, dim=-1)
                logprobs = torch.gather(all_logprobs, 2, responses.unsqueeze(-1)).squeeze(-1)

                output, _ = get_reward(untrained_model, query_responses, tokenizer)
                logits = output.logits[:, context_length - 1 : -1]
                logits /= args.task.temperature
                all_logprobs = F.log_softmax(logits, dim=-1)
                ref_logprobs = torch.gather(all_logprobs, 2, responses.unsqueeze(-1)).squeeze(-1)

                print(f"global_step {global_step}:")
                kl = logprobs - ref_logprobs
                console.print(
                    f"[green]{tokenizer.decode(queries[0], skip_special_tokens=True)}[/]"
                    f"\n[blue]{tokenizer.decode(responses[0], skip_special_tokens=True)}[/]"
                    f"\n[red]reward: {reward[0].item()}[/]"
                    f"\n[red]kl: {kl[0].sum().item()}[/]"
                    f"\n[red]average kl: {kl.sum(1).mean().item()}[/]"
                )
                writer.add_scalar("train/kl", kl.sum(1).mean().item(), global_step)

            # eval on test_label
            test_accuracies = []
            all_inds = np.arange(len(label))
            for start in range(args.labels.num_train, len(label), args.batch_size):
                end = start + args.batch_size
                b_inds_all = all_inds[start:end]
                b_inds = b_inds_all[accelerator.process_index :: accelerator.num_processes]  #  multi-GPU slicing
                mb_data = label[b_inds]
                # print("accelerator.process_index", accelerator.process_index, b_inds, b_inds_all)
                mb_query = torch.from_numpy(np.stack(mb_data["query"]))
                mb_query = left_padding_to_right_padding(mb_query, tokenizer.pad_token_id).to(device)
                mb_best = torch.from_numpy(np.stack(mb_data["best"])).to(device)
                mb_responses = [
                    torch.from_numpy(np.stack(mb_data[f"sample{i}"])).to(device) for i in range(args.labels.num_labels)
                ]
                # hack: deal with openai's padding token
                # assert (mb_query == tokenizer.pad_token_id).sum() == 0
                mb_query[mb_query == OPENAI_PAD_TOKEN_ID] = tokenizer.pad_token_id
                for item in mb_responses:
                    # assert (item == tokenizer.pad_token_id).sum() == 0
                    item[item == OPENAI_PAD_TOKEN_ID] = tokenizer.pad_token_id

                predicted_rewards = []
                for i in range(args.labels.num_labels):
                    query_responses = torch.cat([mb_query, mb_responses[i]], dim=1)
                    if i == 0:
                        print(tokenizer.decode(query_responses[0], skip_special_tokens=True))
                        print(tokenizer.decode(mb_responses[i], skip_special_tokens=True))
                        breakpoint()
                    reward = get_reward(reward_model, query_responses, tokenizer)[1]
                    predicted_rewards.append(reward.squeeze())
                predicted_rewards = torch.stack(
                    predicted_rewards, dim=1
                )  # shape (batch_size, num_labels), basically a reward prediction for each label
                accuracy = (predicted_rewards.argmax(1) == mb_best).float().mean()
                test_accuracies.append(accuracy)
            test_accuracy = accelerator.gather(torch.stack(test_accuracies).mean()).mean().item()
            writer.add_scalar("test/accuracy", test_accuracy, global_step)
            if accelerator.is_main_process:
                print("test/accuracy", test_accuracy, global_step)

    torch.cuda.empty_cache()
    if args.normalize_after:
        normalize(
            args,
            accelerator,
            device,
            tokenizer,
            accelerator.unwrap_model(reward_model).pretrained_model,
            reward_model,
            iter_dataloader,
            generation_config,
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
