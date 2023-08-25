""" Train jax-based reward model for LM human preference details."""
import os
from dataclasses import asdict, dataclass, field
from typing import Optional
import time
import functools
import numpy as np
from torch.utils.data import DataLoader, IterableDataset
import jax
import jax.numpy as jnp
import orbax
import optax
from einops import rearrange
import tyro
from datasets import load_dataset
from rich.pretty import pprint
import transformers
import flax
from flax.training import common_utils
from flax.training import orbax_utils
from flax.core.frozen_dict import freeze
from flax import traverse_util, jax_utils
from flax.training.train_state import TrainState
import flax.linen as nn

# from flax.metrics import tensorboard
# import tensorflow as tf
from lm_human_preference_details.data import DATASET
from torch.utils import tensorboard

# tf.config.experimental.set_visible_devices([], "GPU")


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
    seed: int = 42  # 1
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
    """the name of the dataset to use for labels in
    `https://huggingface.co/datasets/vwxyzjn/lm-human-preferences`"""
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
    rollout_batch_size: int = 512  # decrease this to e.g. 64 if OOM
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
    """Whether, before training, to normalize the rewards on the policy to the
    scales on the training buffer. (For comparisons, just use mean 0, var 1.)"""
    normalize_after: bool = True
    """Whether, after training, to normalize the rewards on the ref policy to
    mean 0, var 1 (so the KL coefficient always has the same meaning)."""
    print_sample_output_freq: int = 10
    """How often to print sample output"""
    save_path: str = "models/"
    """Where to save the model"""
    use_tensorflow_adam: bool = False
    """Whether to use tensorflow-style Adam optimizer instead of PyTorch's"""
    task: TaskHParams = field(default_factory=TaskHParams)
    labels: LabelHParams = field(default_factory=LabelHParams)


OPENAI_PAD_TOKEN_ID = 50259


@flax.struct.dataclass
class RewardModelParams:
    """Parameters for the reward model."""

    backbone_params: flax.core.FrozenDict
    head_params: flax.core.FrozenDict


class RewardHead(nn.Module):
    """Affine transform head for the reward model.

    Attributes:
      head_input_size: Size of the input to the head. The weights of the linear
        layer are initialized from mean-zero Gaussians with std
        1/sqrt(head_input_size + 1).

    Example:
      model = RewardHead(head_input_size=768)
      variables = model.init(jax.random.PRNGKey(0),
        jnp.ones((1, 6, 768)))
    """

    head_input_size: int

    def setup(self):
        self.reward_linear = nn.Dense(
            1,
            kernel_init=nn.initializers.normal(
                stddev=1 / np.sqrt(self.head_input_size + 1)
            ),
            bias_init=nn.initializers.zeros_init(),
        )
        self.reward_gain = self.param("reward_gain", nn.initializers.ones_init(), ())
        self.reward_bias = self.param("reward_bias", nn.initializers.zeros_init(), ())

    def __call__(self, x):
        assert x.shape[-1] == self.head_input_size
        x = self.reward_linear(x)
        x = x * self.reward_gain + self.reward_bias
        return x


class MyDataset(IterableDataset):
    """A dataset for reward model normalization."""

    def __init__(
        self, generator, tokenizer, query_length, seed, start_text=None, end_text=None
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


def right_padding_to_left_padding(tokens, pad_id):
    """Convert from right padding to left padding."""
    assert tokens.ndim == 2
    return np.array(
        [
            [pad_id] * (row == pad_id).sum() + [x for x in row if x != pad_id]
            for row in tokens
        ]
    )


def ceil_div(a, b):
    return (a - 1) // b + 1


def exact_div(a, b):
    q = a // b
    if a != q * b:
        raise ValueError(f"Inexact division: {a} / {b} = {a/b}")
    return q


# TODO: pmap `generate` to accelerate reward model normalization?
def generate(pretrained_model, queries, args, generation_config):
    """generate in a way that does not affect padding tokens"""
    context_length = queries.shape[1]
    attention_mask = queries != args.pad_token_id
    # set padding tokens to 0
    input_ids = jnp.where(attention_mask, queries, 0)
    output = pretrained_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask.astype("int32"),
        # position_ids=attention_mask.cumsum(1) - attention_mask.long(),
        # generation collapsed if this was turned on.
        # TODO: why does generation collapse with this?
        generation_config=generation_config,
        return_dict_in_generate=True,
    )
    return jnp.concatenate((queries, output.sequences[:, context_length:]), axis=1)


def single_epoch_linear_schedule(global_step, args):
    """ anneal learning rate linearly to reach 0 after one epoch."""
    frac = 1.0 - global_step * args.batch_size / args.labels.num_train
    return args.lr * frac


def create_initial_reward_state_and_models(init_key, args):
    # pylint: disable=redefined-outer-name
    """reate reward model and initial reward state."""

    reward_backbone = transformers.FlaxAutoModelForCausalLM.from_pretrained(
        args.base_model
    )
    reward_head = RewardHead(head_input_size=reward_backbone.config.hidden_size)

    if args.use_tensorflow_adam:
        raise NotImplementedError("tensorflow adam is not implemented yet.")
    else:
        optimizer = optax.adam(
            learning_rate=functools.partial(single_epoch_linear_schedule, args=args),
            eps=args.eps,
        )

    if args.gradient_accumulation_steps > 1:
        optimizer = optax.MultiSteps(optimizer, args.gradient_accumulation_steps)
    state = TrainState.create(
        apply_fn=None,
        params=RewardModelParams(
            backbone_params=flax.core.FrozenDict({"params": reward_backbone.params}),
            head_params=flax.core.FrozenDict(
                reward_head.init(
                    init_key,
                    jnp.ones(reward_backbone.config.hidden_size)[None, None, :],
                )
            ),
        ),
        tx=optimizer,
    )
    return state, reward_backbone, reward_head


def get_reward(
    params: RewardModelParams,
    reward_backbone,
    reward_head,
    query_responses_ids: jnp.ndarray,
    args: Args,
):
    """Get reward for each queiry--response pair."""
    assert query_responses_ids.ndim == 2
    # query_responses_ids: [batch_size, length]

    # mask out padding tokens
    attention_mask = query_responses_ids != args.pad_token_id
    query_responses_ids = jnp.where(attention_mask, query_responses_ids, 0)

    # assign position ids
    position_ids = attention_mask.cumsum(1) - attention_mask

    reward_latents = reward_backbone.module.apply(
        variables=params.backbone_params,
        input_ids=query_responses_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        output_hidden_states=True,
    ).hidden_states[-1]
    # shape: [batch_size, length, hidden_size]

    last_reward_latents = reward_latents[:, -1, :]
    # shape: [batch_size, hidden_size]

    reward = reward_head.apply(variables=params.head_params, x=last_reward_latents)
    # shape: [batch_size, 1]
    return reward


def set_reward_state_head_params(
    reward_state: TrainState, gain: float = 1.0, bias: float = 0.0
):
    """Set gain and bias of the reward head.
    Args:
      reward_state: Reward state.
      gain: Gain of the reward head.
      bias: Bias of the reward head.

    Example:
      reward_state = set_reward_state_head_params(
          reward_state, gain=0.1, bias=0.2)
      print(reward_state.params.head_params['params'])
    """
    flat_head_params = traverse_util.flatten_dict(
        reward_state.params.head_params, sep="/"
    )

    flat_head_params["params/reward_gain"] = jnp.array(gain, dtype=jnp.float32)
    flat_head_params["params/reward_bias"] = jnp.array(bias, dtype=jnp.float32)

    unflat_head_params = freeze(traverse_util.unflatten_dict(flat_head_params, sep="/"))

    reward_state = reward_state.replace(
        params=RewardModelParams(
            backbone_params=reward_state.params.backbone_params,
            head_params=unflat_head_params,
        )
    )
    return reward_state


def normalize(
    args,
    tokenizer,
    pretrained_model,
    reward_state,
    iter_dataloader,
    generation_config,
    reward_backbone,
    reward_head,
):
    # number of minibatches for computing the normalization statistics
    n_batches = ceil_div(args.local_normalize_samples, args.rollout_batch_size)

    # reset reward scales
    reward_state = set_reward_state_head_params(reward_state, gain=1.0, bias=0.0)

    def get_normalization_stats(reward_state):
        """compute mean and std of rewards"""

        sample_queries_responses = []
        for _ in range(n_batches):
            data = next(iter_dataloader)
            queries = data["input_ids"]
            queries = right_padding_to_left_padding(
                data["input_ids"], args.pad_token_id
            )
            query_responses = generate(
                pretrained_model, queries, tokenizer, generation_config
            )
            sample_queries_responses.append(query_responses)

        rewards = []
        for query_responses in sample_queries_responses:
            rewards.append(
                get_reward(
                    reward_state.params,
                    reward_backbone,
                    reward_head,
                    query_responses,
                    args,
                )
            )
        # Here, len(rewards) = n_batches
        # each rewards[i] is a (args.rollout_batch_size, 1) array.

        rewards = np.concatenate(rewards)
        # rewards shape: [args.local_normalize_samples, 1]
        mean, std = rewards.mean(), rewards.std()
        print(f"mean: {mean}, std: {std}")
        return mean, std

    mean, std = get_normalization_stats(reward_state)
    target_mean, target_std = 0.0, 1.0
    gain = target_std / std
    bias = target_mean - gain * mean
    print(f"gain: {gain}, bias: {bias}")

    # do normalization
    reward_state = set_reward_state_head_params(reward_state, gain=gain, bias=bias)

    # validate normalization
    _, _ = get_normalization_stats(reward_state)
    return reward_state


def prepare_left_padded_query_responses_with_labels(dataset, args):
    """Prepare left padded, concatenated queries and responses, and add labels.
    Args:
      dataset: a dictionary that contains 'query', 'best', and 'sample{i}',
        where i is from 0 to args.labels.num_labels-1.
      args: a dataclass that contains 'labels.num_labels' and 'pad_token_id'.

    Returns:
      queries_responses: array of concatenated queries and responses, with shape
        [num_queires, num_responses_per_query, max_query_len + max_response_len]
      labels:
        array of the best response idx for each label, with shape
        [num_queires, 1]
    """

    labels = np.array(dataset["best"])
    # [num_queires,]

    queries = np.stack(dataset["query"])
    # [num_queires, max_query_length]

    queries = np.repeat(queries, args.labels.num_labels, axis=0)
    queries = rearrange(queries, "(q r) l -> q r l", r=args.labels.num_labels)
    # [num_queires, num_queires, max_query_length]

    responses = np.array(
        [np.stack(dataset[f"sample{i}"]) for i in range(args.labels.num_labels)]
    )
    # [num_response_per_query, num_queires, max_response_len]

    responses = rearrange(responses, "r q l -> q r l")
    # [num_queires, num_responses_per_query, max_response_len]

    queries_responses = np.concatenate([queries, responses], axis=-1)
    # [num_queires, num_responses_per_query, max_query_length + max_response_len]

    queries_responses[queries_responses == OPENAI_PAD_TOKEN_ID] = args.pad_token_id

    queries_responses = right_padding_to_left_padding(
        rearrange(queries_responses, "q r l -> (q r) l"), pad_id=args.pad_token_id,
    )

    queries_responses = rearrange(
        queries_responses, "(q r) l -> q r l", r=args.labels.num_labels
    )
    # [num_queires, num_responses_per_query, max_query_len + max_response_len]
    return queries_responses, labels


def get_dataloader_iter(rng, dataset_tokens, dataset_labels, args):
    """Get iteration of dataloader."""
    assert dataset_tokens.shape[0] == dataset_labels.shape[0]
    num_samples = dataset_tokens.shape[0]

    steps_per_epoch = num_samples // args.batch_size
    perms = jax.random.permutation(rng, num_samples)
    # Skip incomplete batch:
    perms = perms[: steps_per_epoch * args.batch_size]
    perms = perms.reshape((steps_per_epoch, args.batch_size))

    for perm in perms:
        batch = (dataset_tokens[perm], dataset_labels[perm])
        yield batch


def train_step(state, batch, reward_backbone, reward_head, args):
    """Train reward model for one step."""
    query_responses, labels = batch
    query_responses_ids = rearrange(query_responses, "q r l -> (q r) l")
    # query_responses_ids: [num_queries * num_responses_per_query, length]

    def loss_function(params):
        logits = get_reward(
            params, reward_backbone, reward_head, query_responses_ids, args
        )

        logits_reshaped = rearrange(logits, "(q r) 1 -> q r", r=args.labels.num_labels)

        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits_reshaped, labels
        ).mean()

        accuracy = (logits_reshaped.argmax(axis=1) == labels).astype("float32").mean()
        return loss, accuracy

    loss_grad_fn = jax.value_and_grad(loss_function, has_aux=True)
    (loss, accuracy), grads = loss_grad_fn(state.params)
    grads = jax.lax.pmean(grads, "batch")
    state = state.apply_gradients(grads=grads)
    loss = jax.lax.pmean(loss, axis_name="batch")
    accuracy = jax.lax.pmean(accuracy, axis_name="batch")
    return state, {"loss": loss, "accuracy": accuracy}


def val_step(state, batch, reward_backbone, reward_head, args):
    """Eval reward model for one step."""
    query_responses, labels = batch
    query_responses_ids = rearrange(query_responses, "q r l -> (q r) l")
    # query_responses_ids: [num_queries * num_responses_per_query, length]

    def loss_function(params):
        logits = get_reward(
            params, reward_backbone, reward_head, query_responses_ids, args
        )

        logits_reshaped = rearrange(logits, "(q r) 1 -> q r", r=args.labels.num_labels)

        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits_reshaped, labels
        ).mean()

        accuracy = (logits_reshaped.argmax(axis=1) == labels).astype("float32").mean()
        return loss, accuracy

    loss, accuracy = loss_function(state.params)
    loss = jax.lax.pmean(loss, axis_name="batch")
    accuracy = jax.lax.pmean(accuracy, axis_name="batch")
    return {"loss": loss, "accuracy": accuracy}


def train(args: Args):
    args.world_size = len(jax.devices())

    args.batch_size = int(args.local_batch_size * args.world_size)
    args.normalize_samples = int(args.local_normalize_samples * args.world_size)
    args.local_micro_batch_size = exact_div(
        args.local_batch_size, args.gradient_accumulation_steps
    )

    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"

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

    writer = tensorboard.SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    pprint(args)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.base_model, padding_side="right",
    )
    # we use the padding token manually but do not resize the token embedding of the model
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    args.pad_token_id = tokenizer.pad_token_id

    untrained_model = transformers.FlaxAutoModelForCausalLM.from_pretrained(
        args.base_model
    )

    reward_state, reward_backbone, reward_head = create_initial_reward_state_and_models(
        jax.random.PRNGKey(args.seed), args
    )

    p_train_step = jax.pmap(
        functools.partial(
            train_step,
            args=args,
            reward_backbone=reward_backbone,
            reward_head=reward_head,
        ),
        axis_name="batch",
        donate_argnums=(0,),
    )
    p_val_step = jax.pmap(
        functools.partial(
            val_step,
            args=args,
            reward_backbone=reward_backbone,
            reward_head=reward_head,
        ),
        axis_name="batch",
    )

    normalization_dataset = MyDataset(
        DATASET[args.task.query_dataset],
        tokenizer,
        args.task.query_length,
        seed=args.seed,
        start_text=args.task.start_text,
        end_text=args.task.end_text,
    )
    normalization_dataloader = DataLoader(
        normalization_dataset, batch_size=args.rollout_batch_size
    )
    iter_normalization_dataloader = iter(normalization_dataloader)

    generation_config = transformers.GenerationConfig(
        max_new_tokens=args.task.response_length,
        min_new_tokens=args.task.response_length,
        temperature=args.task.temperature,
        top_k=0.0,
        top_p=1.0,
        do_sample=True,
        pad_token_id=args.pad_token_id,
    )

    if args.normalize_before:
        print("===Normalize reward model *before* training===")

        # pylint: disable=E1101:no-member
        print(
            "before normalization. "
            + f"Gain: {reward_state.params.head_params['params']['reward_gain']}"
            + f" Bias: {reward_state.params.head_params['params']['reward_bias']}"
        )

        reward_state = normalize(
            args,
            tokenizer,
            untrained_model,
            reward_state,
            iter_normalization_dataloader,
            generation_config,
            reward_backbone,
            reward_head,
        )

        print(
            "after normalization. "
            + f"Gain: {reward_state.params.head_params['params']['reward_gain']}"
            + f" Bias: {reward_state.params.head_params['params']['reward_bias']}"
        )

    reward_state = jax_utils.replicate(reward_state)

    # `labeled_dataset` has keys
    # `['sample0', 'query', 'best', 'sample3', 'sample1', 'sample2']`
    labeled_dataset = load_dataset(
        "vwxyzjn/lm-human-preferences", data_files=[args.label_dataset],
    )["train"]
    print("Num labels found in source:", len(labeled_dataset))
    print("training on", args.labels.num_train, "in batches of", args.local_batch_size)

    all_queries_responses, all_labels = prepare_left_padded_query_responses_with_labels(
        labeled_dataset, args
    )

    assert args.labels.num_train < all_queries_responses.shape[0]
    train_queries_responses = all_queries_responses[: args.labels.num_train]
    train_labels = all_labels[: args.labels.num_train]

    val_queries_responses = all_queries_responses[args.labels.num_train :]
    val_labels = all_labels[args.labels.num_train :]

    train_iter = get_dataloader_iter(
        jax.random.PRNGKey(args.seed),
        dataset_tokens=train_queries_responses,
        dataset_labels=train_labels,
        args=args,
    )

    print("===training reward model===")

    for global_step, train_batch in enumerate(train_iter):
        train_batch = common_utils.shard(train_batch)
        reward_state, train_metrics = p_train_step(reward_state, train_batch)
        writer.add_scalar(
            "train/lr", single_epoch_linear_schedule(global_step, args), global_step
        )

        # gathering replicated metric data
        train_metrics = common_utils.get_metrics([train_metrics])

        for key, value in train_metrics.items():
            writer.add_scalar(f"train/{key}", value, global_step)

        if (
            args.print_sample_output_freq > 0
            and global_step % args.print_sample_output_freq == 0
        ):
            val_iter = get_dataloader_iter(
                jax.random.PRNGKey(0),
                dataset_tokens=val_queries_responses,
                dataset_labels=val_labels,
                args=args,
            )

            val_metrics_list = []
            for val_batch in val_iter:
                val_batch = common_utils.shard(val_batch)
                val_metrics = p_val_step(reward_state, val_batch)
                val_metrics_list.append(val_metrics)

            val_metrics = common_utils.get_metrics(val_metrics_list)
            for key, value in val_metrics.items():
                val_metrics[key] = value.mean()
                writer.add_scalar(f"test/{key}", val_metrics[key], global_step)

            print(
                f"gloabl_step: {global_step} | "
                + f"test/accuracy {val_metrics['accuracy']}"
            )

    reward_state = jax_utils.unreplicate(reward_state)

    if args.normalize_after:
        print("===Normalize reward model *after* training===")

        # pylint: disable=E1101:no-member
        print(
            "before normalization. "
            + f"Gain: {reward_state.params.head_params['params']['reward_gain']}"
            + f" Bias: {reward_state.params.head_params['params']['reward_bias']}"
        )

        reward_state = normalize(
            args,
            tokenizer,
            untrained_model,
            reward_state,
            iter_normalization_dataloader,
            generation_config,
            reward_backbone,
            reward_head,
        )
        print(
            "after normalization. "
            + f"Gain: {reward_state.params.head_params['params']['reward_gain']}"
            + f" Bias: {reward_state.params.head_params['params']['reward_bias']}"
        )

    if args.save_path:
        ckpt = {"reward_model": reward_state, "args": vars(args)}
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(ckpt)
        orbax_checkpointer.save(args.save_path, ckpt, save_args=save_args, force=True)

    if args.track:
        wandb.finish()


if __name__ == "__main__":
    args = tyro.cli(Args)
    train(args)
