import os
import random
import time
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace
from typing import Optional

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, broadcast
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tyro
from rich.console import Console
from datasets import load_dataset
from rich.pretty import pprint
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from lm_human_preference_details.data import DATASET

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
    exp_name: str = os.path.basename(__file__)[:-len(".py")]
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
    gradient_accumulation_steps: int = 1
    """gradient accumulation steps"""
    local_micro_batch_size: tyro.conf.Suppress[int] = None
    """per rank micro batch size"""
    lr: float = 0.00005
    """the learning rate"""
    eps: float = 1e-5
    """the epsilon for AdamW"""
    rollout_batch_size: int = 512
    """rollot batch size"""
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
    print_sample_output_freq: int = 10
    """How often to print sample output"""
    save_path: str = "models/reward.pt"
    """Where to save the model"""
    use_tensorflow_adam: bool = True
    """Whether to use tensorflow-style Adam optimizer instead of PyTorch's"""
    task: TaskHParams = field(default_factory=TaskHParams)
    labels: LabelHParams = field(default_factory=LabelHParams)


from torch import Tensor, optim
from typing import List, Optional
from torch.optim.optimizer import _use_grad_for_differentiable, _get_value, _dispatch_sqrt


def _single_tensor_adam(params: List[Tensor],
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
                        differentiable: bool):

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
        lr_t = lr * _dispatch_sqrt((1 - beta2 ** step)) / (1 - beta1 ** step)
        denom = exp_avg_sq.sqrt().add_(eps)
        param.addcdiv_(exp_avg, denom, value=-lr_t)


def adam(params: List[Tensor],
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
         maximize: bool):
    
    func = _single_tensor_adam

    func(params,
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
         found_inf=found_inf)

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
            beta1, beta2 = group['betas']

            self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps)

            adam(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=group['amsgrad'],
                beta1=beta1,
                beta2=beta2,
                lr=group['lr'],
                weight_decay=group['weight_decay'],
                eps=group['eps'],
                maximize=group['maximize'],
                foreach=group['foreach'],
                capturable=group['capturable'],
                differentiable=group['differentiable'],
                fused=group['fused'],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
            )

        return loss


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.normal_(layer.weight, std=std)
    torch.nn.init.constant_(layer.bias, val=bias_const)
    return layer


OPENAI_PAD_TOKEN_ID = 50259


class AutoModelForCausalLMWithScalarHead(nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.scalar_head = layer_init(nn.Linear(pretrained_model.config.hidden_size, 1), std=0)

    def forward(self, **kwargs):
        output = self.pretrained_model(**kwargs)
        return output, self.scalar_head(output.hidden_states[-1])


class AutoModelForCausalLMWithRewardHead(nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.scalar_head = layer_init(nn.Linear(pretrained_model.config.hidden_size, 1), std=1 / np.sqrt(pretrained_model.config.hidden_size + 1))
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
    def __init__(self, generator, tokenizer, query_length, seed, start_text=None, end_text=None):
        self.generator = generator
        self.tokenizer = tokenizer
        self.query_length = query_length
        self.start_text = start_text
        self.end_text = end_text
        self.seed = seed
        token_to_index = tokenizer.get_vocab()
        self.start_token = token_to_index[start_text] if self.start_text else None
        self.end_token = token_to_index[end_text] if self.end_text else None

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
                return_attention_mask=True,
            )
            yield output


def left_padding_to_right_padding(query, pad_id):
    # got to convert to right padding, otherwise `transformers` has weird issues
    # even with `position_ids`
    return torch.tensor([
        [pad_id]*(row==pad_id).sum() + [x for x in row if x != pad_id]
        for row in query
    ])


def ceil_div(a, b):
    return (a - 1) // b + 1


def exact_div(a, b):
    q = a // b
    if a != q * b:
        raise ValueError('Inexact division: %s / %s = %s' % (a, b, a / b))
    return q


def generate(pretrained_model, queries, tokenizer, generation_config):
    """generate in a way that does not affect padding tokens"""
    context_length = queries.shape[1]
    attention_mask = queries != tokenizer.pad_token_id
    input_ids = queries.clone()
    input_ids[~attention_mask] = 0 # set padding tokens to 0
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
    position_ids = attention_mask.cumsum(1) - attention_mask.long() # exclusive cumsum
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
        n_batches = ceil_div(args.local_normalize_samples, args.rollout_batch_size)
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
        rewards= accelerator.gather(rewards)
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
        n_batches = ceil_div(args.local_normalize_samples, args.rollout_batch_size)
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
        rewards= accelerator.gather(rewards)
        mean, std = rewards.mean(), rewards.std()
        print(f"after mean: {mean}, after std: {std}")


def train(args: Args):
    accelerator = Accelerator(
        kwargs_handlers=[DistributedDataParallelKwargs(broadcast_buffers=False)], # this is needed to avoid https://github.com/pytorch/pytorch/issues/22095#issuecomment-505099500
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    args.world_size = accelerator.num_processes
    args.batch_size = int(args.local_batch_size * args.world_size)
    args.normalize_samples = int(args.local_normalize_samples * args.world_size)
    args.local_micro_batch_size = exact_div(args.local_batch_size, args.gradient_accumulation_steps)

    console = Console(force_terminal=True)
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SimpleNamespace() # dummy writer
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
    # jax-style rng key generation; so that the rng keys are not continuous / sequential (e.g., [1, 2, 3, 4])
    # they should be like [1715945195,  504011663, 1037299162, ...]
    rng = np.random.default_rng(args.seed)
    rng_keys = rng.integers(low=1, high=np.iinfo(np.int32).max, size=(accelerator.num_processes,))
    local_seed = rng_keys[accelerator.process_index]
    random.seed(local_seed)
    np.random.seed(local_seed)
    torch.manual_seed(local_seed)
    torch.backends.cudnn.deterministic = True
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        padding_side="right",
    )
    # we use the padding token manually but do not resize the token embedding of the model
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    untrained_model = AutoModelForCausalLMWithRewardHead(AutoModelForCausalLM.from_pretrained(args.base_model)).to(device)
    reward_model = AutoModelForCausalLMWithRewardHead(AutoModelForCausalLM.from_pretrained(args.base_model)).to(device)
    reward_model.pretrained_model.generation_config.eos_token_id = None  # disable `pad_token_id` and `eos_token_id` because we just want to
    reward_model.pretrained_model.generation_config.pad_token_id = None  # generate tokens without truncation / padding
    if args.use_tensorflow_adam:
        optimizer = AdamTensorFlowStyle(reward_model.parameters(), lr=args.lr, eps=args.eps)
    else:
        optimizer = optim.Adam(reward_model.parameters(), lr=args.lr, eps=args.eps)
    dataset = MyDataset(
        DATASET[args.task.query_dataset],
        tokenizer,
        args.task.query_length,
        seed=local_seed,
        start_text=args.task.start_text,
        end_text=args.task.end_text,
    )
    dataloader = DataLoader(dataset, batch_size=args.rollout_batch_size)
    reward_model, optimizer, dataloader = accelerator.prepare(reward_model, optimizer, dataloader)
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
        normalize(args, accelerator, device, tokenizer, accelerator.unwrap_model(reward_model).pretrained_model, reward_model, iter_dataloader, generation_config)
    print("after====", reward_model.module.reward_gain.data)

    print("===training reward model===")
    all_inds = np.arange(args.labels.num_train)
    np.random.shuffle(all_inds)
    # ensure that all processes have the same shuffled indices
    all_inds = broadcast(torch.tensor(all_inds, device=device), 0)
    all_inds = all_inds.cpu().numpy()
    global_step = 0
    for start in range(0, args.labels.num_train, args.batch_size):
        # linear rate annealing
        lr = (1 - start / args.labels.num_train) * args.lr
        optimizer.param_groups[0]["lr"] = lr

        with accelerator.accumulate(reward_model):
            global_step += 1
            end = start + args.batch_size
            b_inds_all = all_inds[start:end]
            b_inds = b_inds_all[accelerator.process_index::accelerator.num_processes] #  multi-GPU slicing
            losses = torch.zeros((args.gradient_accumulation_steps,), device=device)
            accuracies = torch.zeros((args.gradient_accumulation_steps,), device=device)
            gradient_accumulation_step = 0
            for micro_batch_start in range(0, args.local_batch_size, args.local_micro_batch_size):
                micro_batch_end = micro_batch_start + args.local_micro_batch_size 
                micro_batch_inds = b_inds[micro_batch_start:micro_batch_end]
                mb_data = label[micro_batch_inds]
                mb_query = torch.from_numpy(np.stack(mb_data["query"]))
                mb_query = left_padding_to_right_padding(mb_query, tokenizer.pad_token_id).to(device)
                mb_best = torch.from_numpy(np.stack(mb_data["best"])).to(device)
                mb_responses = [
                    torch.from_numpy(np.stack(mb_data[f"sample{i}"])).to(device)
                    for i in range(args.labels.num_labels)
                ]
                # hack: deal with openai's padding token
                mb_query[mb_query == OPENAI_PAD_TOKEN_ID] = tokenizer.pad_token_id
                for item in mb_responses:
                    item[item == OPENAI_PAD_TOKEN_ID] = tokenizer.pad_token_id

                predicted_rewards = []
                for i in range(args.labels.num_labels):
                    query_responses = torch.cat([mb_query, mb_responses[i]], dim=1)
                    reward = get_reward(reward_model, query_responses, tokenizer)[1]
                    predicted_rewards.append(
                        reward.view(-1)
                    )
                predicted_rewards = torch.stack(
                    predicted_rewards, dim=1
                )  # shape (batch_size, num_labels), basically a reward prediction for each label
                accuracy = (predicted_rewards.argmax(1) == mb_best).float().mean()
                loss = torch.nn.functional.cross_entropy(predicted_rewards, mb_best)
                accelerator.backward(loss)
                optimizer.step()  # accelerate handles gradient accumulation automatically
                optimizer.zero_grad()
                losses[gradient_accumulation_step] = loss
                accuracies[gradient_accumulation_step] = accuracy
                gradient_accumulation_step += 1

        writer.add_scalar("train/loss", accelerator.gather(losses).mean().item(), global_step)
        writer.add_scalar("train/accuracy", accelerator.gather(accuracies).mean().item(), global_step)
        writer.add_scalar("train/lr", lr, global_step)

        if args.print_sample_output_freq > 0 and global_step % args.print_sample_output_freq == 0:
            with torch.no_grad():
                # eval on test_label, some duplicate code (I don't want to make the training loop into a function...)
                test_accuracies = []
                all_inds = np.arange(len(label))
                for start in range(args.labels.num_train, len(label), args.batch_size):
                    end = start + args.batch_size
                    b_inds_all = all_inds[start:end]
                    b_inds = b_inds_all[accelerator.process_index::accelerator.num_processes] #  multi-GPU slicing
                    for micro_batch_start in range(0, args.local_batch_size, args.local_micro_batch_size):
                        micro_batch_end = micro_batch_start + args.local_micro_batch_size 
                        micro_batch_inds = b_inds[micro_batch_start:micro_batch_end]
                        mb_data = label[micro_batch_inds]
                        mb_query = torch.from_numpy(np.stack(mb_data["query"]))
                        mb_query = left_padding_to_right_padding(mb_query, tokenizer.pad_token_id).to(device)
                        mb_best = torch.from_numpy(np.stack(mb_data["best"])).to(device)
                        mb_responses = [
                            torch.from_numpy(np.stack(mb_data[f"sample{i}"])).to(device)
                            for i in range(args.labels.num_labels)
                        ]
                        # hack: deal with openai's padding token
                        mb_query[mb_query == OPENAI_PAD_TOKEN_ID] = tokenizer.pad_token_id
                        for item in mb_responses:
                            item[item == OPENAI_PAD_TOKEN_ID] = tokenizer.pad_token_id
                        predicted_rewards = []
                        for i in range(args.labels.num_labels):
                            query_responses = torch.cat([mb_query, mb_responses[i]], dim=1)
                            reward = get_reward(reward_model, query_responses, tokenizer)[1]
                            predicted_rewards.append(
                                reward.view(-1)
                            )
                        predicted_rewards = torch.stack(
                            predicted_rewards, dim=1
                        )  # shape (batch_size, num_labels), basically a reward prediction for each label
                        accuracy = (predicted_rewards.argmax(1) == mb_best).float().mean()
                        test_accuracies.append(accuracy)
                test_accuracy = accelerator.gather(torch.stack(test_accuracies).mean()).mean().item()
                writer.add_scalar("test/accuracy", test_accuracy, global_step)
                if accelerator.is_main_process:
                    print("test/accuracy", test_accuracy, global_step)

                # the part below is testing out some generations and KLs, not presented in the original code
                data = next(iter_dataloader)
                queries = data["input_ids"].to(device)
                context_length = queries.shape[1]
                queries = left_padding_to_right_padding(data["input_ids"], tokenizer.pad_token_id).to(device)
                query_responses = generate(accelerator.unwrap_model(reward_model).pretrained_model, queries, tokenizer, generation_config)
                responses = query_responses[:, context_length:]

                output, reward = get_reward(reward_model, query_responses, tokenizer)
                logits = output.logits[:,context_length-1:-1]
                logits /= args.task.temperature
                all_logprobs = F.log_softmax(logits, dim=-1)
                logprobs = torch.gather(all_logprobs, 2, responses.unsqueeze(-1)).squeeze(-1)
                del output, logits, all_logprobs; torch.cuda.empty_cache()

                output, _ = get_reward(untrained_model, query_responses, tokenizer)
                logits = output.logits[:,context_length-1:-1]
                logits /= args.task.temperature
                all_logprobs = F.log_softmax(logits, dim=-1)
                ref_logprobs = torch.gather(all_logprobs, 2, responses.unsqueeze(-1)).squeeze(-1)
                del output, logits, all_logprobs; torch.cuda.empty_cache()

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

    torch.cuda.empty_cache()
    if args.normalize_after:
        normalize(args, accelerator, device, tokenizer, accelerator.unwrap_model(reward_model).pretrained_model, reward_model, iter_dataloader, generation_config)

    # save model
    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        torch.save(accelerator.unwrap_model(reward_model).state_dict(), args.save_path)

    if accelerator.is_main_process and args.track:
        wandb.finish()


if __name__ == "__main__":
    args = tyro.cli(Args)
    train(args)
