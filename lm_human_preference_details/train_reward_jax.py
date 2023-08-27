import functools
import os
import time
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace
from typing import List, Optional

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax
import tyro
from datasets import load_dataset
from einops import rearrange
from flax import jax_utils, traverse_util
from flax.core.frozen_dict import freeze
from flax.training import common_utils, orbax_utils
from flax.training.train_state import TrainState
from optax import ScaleByAdamState, update_moment, update_moment_per_elem_norm
from optax._src import base, combine, numerics, utils
from optax._src.alias import _scale_by_learning_rate
from rich.pretty import pprint
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, FlaxAutoModelForCausalLM, GenerationConfig

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
    distributed: bool = False
    "whether to use `jax.distirbuted`"
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
    print_sample_output_freq: int = 10
    """How often to print sample output"""
    save_path: str = "models/reward.pt"
    """Where to save the model"""
    use_tensorflow_adam: bool = True
    """Whether to use tensorflow-style Adam optimizer instead of PyTorch's"""

    # distributed settings
    local_rank: int = 0
    """the rank of this process"""
    learner_device_ids: List[int] = field(default_factory=lambda: [0])
    "the device ids that script will use"
    learner_devices: tyro.conf.Suppress[int] = None # real type is `List[str]`
    """the devices that script will use"""
    global_learner_decices: tyro.conf.Suppress[int] = None # real type is `List[str]`
    """the total devices (across all nodes and machines) that script will use"""
    task: TaskHParams = field(default_factory=TaskHParams)
    labels: LabelHParams = field(default_factory=LabelHParams)


OPENAI_PAD_TOKEN_ID = 50259


def scale_by_adam_tf_style(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype=None,
) -> base.GradientTransformation:
    """Rescale updates according to the Adam algorithm.
    References:
            [Kingma et al, 2014](https://arxiv.org/abs/1412.6980)
    Args:
            b1: Decay rate for the exponentially weighted average of grads.
            b2: Decay rate for the exponentially weighted average of squared grads.
            eps: Term added to the denominator to improve numerical stability.
            eps_root: Term added to the denominator inside the square-root to improve
                    numerical stability when backpropagating gradients through the rescaling.
            mu_dtype: Optional `dtype` to be used for the first order accumulator; if
                    `None` then the `dtype` is inferred from `params` and `updates`.
    Returns:
            A `GradientTransformation` object.
    """

    mu_dtype = utils.canonicalize_dtype(mu_dtype)

    def init_fn(params):
        mu = jax.tree_util.tree_map(lambda t: jnp.zeros_like(t, dtype=mu_dtype), params)  # First moment
        nu = jax.tree_util.tree_map(jnp.zeros_like, params)  # Second moment
        return ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

    def update_fn(updates, state, params=None):
        del params
        mu = update_moment(updates, state.mu, b1, 1)
        nu = update_moment_per_elem_norm(updates, state.nu, b2, 2)
        count_inc = numerics.safe_int32_increment(state.count)

        ### `optax` default adam implementation
        # mu_hat = bias_correction(mu, b1, count_inc)
        # nu_hat = bias_correction(nu, b2, count_inc)
        # updates = jax.tree_util.tree_map(
        #     lambda m, v: m / (jnp.sqrt(v + eps_root) + eps), mu_hat, nu_hat)
        ### Tensorflow adam implementation
        updates = jax.tree_util.tree_map(
            lambda m, v: (jnp.sqrt(1 - b2**count_inc) / (1 - b1**count_inc)) * m / (jnp.sqrt(v + eps_root) + eps),
            mu,
            nu,
        )  #
        mu = utils.cast_tree(mu, mu_dtype)
        return updates, ScaleByAdamState(count=count_inc, mu=mu, nu=nu)

    return base.GradientTransformation(init_fn, update_fn)


def adam_tf_style(
    learning_rate,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype=None,
):
    return combine.chain(
        scale_by_adam_tf_style(b1=b1, b2=b2, eps=eps, eps_root=eps_root, mu_dtype=mu_dtype),
        _scale_by_learning_rate(learning_rate),
    )


@flax.struct.dataclass
class RewardModelParams:
    """Parameters for the reward model."""

    lm_backbone_params: flax.core.FrozenDict
    head_params: flax.core.FrozenDict


class RewardHead(nn.Module):
    head_input_size: int

    @nn.compact
    def __call__(self, x):
        assert x.shape[-1] == self.head_input_size
        x = nn.Dense(
            1,
            kernel_init=nn.initializers.normal(stddev=1 / np.sqrt(self.head_input_size + 1)),
            bias_init=nn.initializers.zeros_init(),
        )(x)
        reward_gain = self.param("reward_gain", nn.initializers.ones_init(), ())
        reward_bias = self.param("reward_bias", nn.initializers.zeros_init(), ())
        x = x * reward_gain + reward_bias
        return x


# Dataset for reward-model normalization
class NormalizationDataset(IterableDataset):
    """A dataset for reward model normalization."""

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


def right_padding_to_left_padding(tokens, pad_id):
    """Convert from right padding to left padding."""
    assert tokens.ndim == 2
    return np.array([[pad_id] * (row == pad_id).sum() + [x for x in row if x != pad_id] for row in tokens])


def ceil_div(a, b):
    return (a - 1) // b + 1


def exact_div(a, b):
    q = a // b
    if a != q * b:
        raise ValueError(f"Inexact division: {a} / {b} = {a / b}")
    return q


# TODO: pmap `generate` to accelerate reward model normalization?
def generate(lm_backbone, queries, args, generation_config):
    """generate in a way that does not affect padding tokens"""
    context_length = queries.shape[1]
    attention_mask = queries != args.pad_token_id
    input_ids = jnp.where(attention_mask, queries, 0)  # set padding tokens to 0
    output = lm_backbone.generate(
        input_ids=input_ids,
        attention_mask=attention_mask.astype("int32"),
        # need to convert to int for now, due to the bug https://github.com/huggingface/transformers/issues/25634
        # position_ids=attention_mask.cumsum(1) - attention_mask.long(), # generation collapsed if this was turned on. TODO: why does generation collapse with this?
        generation_config=generation_config,
        return_dict_in_generate=True,
    )
    # restore padding tokens
    return jnp.concatenate((queries, output.sequences[:, context_length:]), axis=1)


def single_epoch_linear_schedule(global_step, args):
    """anneal learning rate linearly to reach 0 after one epoch."""
    frac = 1.0 - global_step * args.batch_size / args.labels.num_train
    return args.lr * frac


def create_initial_reward_state_and_models(init_key, args):
    """reate reward model and initial reward state."""

    lm_backbone = FlaxAutoModelForCausalLM.from_pretrained(args.base_model)
    scalar_head = RewardHead(head_input_size=lm_backbone.config.hidden_size)

    def get_reward(
        params: RewardModelParams,
        query_responses_ids: jnp.ndarray,
    ):
        """Get reward for each queiry--response pair."""
        assert query_responses_ids.ndim == 2
        # query_responses_ids: [batch_size, length]

        # mask out padding tokens
        attention_mask = query_responses_ids != args.pad_token_id
        query_responses_ids = jnp.where(attention_mask, query_responses_ids, 0)

        # assign position ids
        position_ids = attention_mask.cumsum(1) - attention_mask

        reward_latents = lm_backbone.module.apply(
            variables=params.lm_backbone_params,
            input_ids=query_responses_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
        ).hidden_states[-1]
        # shape: [batch_size, length, hidden_size]

        last_reward_latents = reward_latents[:, -1, :]
        # shape: [batch_size, hidden_size]

        reward = scalar_head.apply(variables=params.head_params, x=last_reward_latents)
        # shape: [batch_size, 1]
        return reward

    if args.use_tensorflow_adam:
        adam = adam_tf_style
    else:
        adam = optax.adam

    optimizer = adam(
        learning_rate=functools.partial(single_epoch_linear_schedule, args=args),
        eps=args.eps,
    )

    optimizer = optax.MultiSteps(optimizer, args.gradient_accumulation_steps)
    state = TrainState.create(
        apply_fn=get_reward,
        params=RewardModelParams(
            lm_backbone_params=flax.core.FrozenDict({"params": lm_backbone.params}),
            head_params=flax.core.FrozenDict(
                scalar_head.init(
                    init_key,
                    jnp.ones(lm_backbone.config.hidden_size)[None, None, :],
                )
            ),
        ),
        tx=optimizer,
    )
    return state


def set_reward_state_head_params(reward_state: TrainState, gain: float = 1.0, bias: float = 0.0):
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
    flat_head_params = traverse_util.flatten_dict(reward_state.params.head_params, sep="/")

    flat_head_params["params/reward_gain"] = jnp.array(gain, dtype=jnp.float32)
    flat_head_params["params/reward_bias"] = jnp.array(bias, dtype=jnp.float32)

    unflat_head_params = freeze(traverse_util.unflatten_dict(flat_head_params, sep="/"))

    reward_state = reward_state.replace(
        params=RewardModelParams(
            lm_backbone_params=reward_state.params.lm_backbone_params,
            head_params=unflat_head_params,
        )
    )
    return reward_state


def normalize(
    args,
    lm_backbone,
    reward_state,
    iter_dataloader,
    generation_config,
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
            queries = right_padding_to_left_padding(data["input_ids"], args.pad_token_id)
            query_responses = generate(lm_backbone, queries, args, generation_config)
            sample_queries_responses.append(query_responses)

        rewards = []
        for query_responses in sample_queries_responses:
            rewards.append(
                reward_state.apply_fn(
                    reward_state.params,
                    query_responses,
                )
            )
        # Here, len(rewards) = n_batches
        # each rewards[i] is a (args.rollout_batch_size, 1) array.

        rewards = np.concatenate(rewards)
        # shape: [args.local_normalize_samples, 1]
        mean, std = rewards.mean(), rewards.std()
        print(f"mean: {mean}, std: {std}")
        return mean, std

    # reward normalization
    mean, std = get_normalization_stats(reward_state)
    target_mean, target_std = 0.0, 1.0
    gain = target_std / std
    bias = target_mean - gain * mean
    print(f"gain: {gain}, bias: {bias}")
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

    responses = np.array([np.stack(dataset[f"sample{i}"]) for i in range(args.labels.num_labels)])
    # [num_response_per_query, num_queires, max_response_len]

    responses = rearrange(responses, "r q l -> q r l")
    # [num_queires, num_responses_per_query, max_response_len]

    queries_responses = np.concatenate([queries, responses], axis=-1)
    # [num_queires, num_responses_per_query, max_query_length + max_response_len]

    queries_responses[queries_responses == OPENAI_PAD_TOKEN_ID] = args.pad_token_id

    queries_responses = right_padding_to_left_padding(
        rearrange(queries_responses, "q r l -> (q r) l"),
        pad_id=args.pad_token_id,
    )

    queries_responses = rearrange(queries_responses, "(q r) l -> q r l", r=args.labels.num_labels)
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


def train_step(state, batch, args):
    """Train reward model for one step."""
    query_responses, labels = batch
    query_responses_ids = rearrange(query_responses, "q r l -> (q r) l")
    # shape: [num_queries * num_responses_per_query, length]

    def loss_function(params):
        logits = state.apply_fn(params, query_responses_ids)
        logits_reshaped = rearrange(logits, "(q r) 1 -> q r", r=args.labels.num_labels)

        loss = optax.softmax_cross_entropy_with_integer_labels(logits_reshaped, labels).mean()

        accuracy = (logits_reshaped.argmax(axis=1) == labels).astype("float32").mean()
        return loss, accuracy

    loss_grad_fn = jax.value_and_grad(loss_function, has_aux=True)
    (loss, accuracy), grads = loss_grad_fn(state.params)
    grads = jax.lax.pmean(grads, "batch")
    state = state.apply_gradients(grads=grads)
    loss = jax.lax.pmean(loss, axis_name="batch")
    accuracy = jax.lax.pmean(accuracy, axis_name="batch")
    return state, {"loss": loss, "accuracy": accuracy}


def val_step(state, batch, args):
    """Eval reward model for one step."""
    query_responses, labels = batch
    query_responses_ids = rearrange(query_responses, "q r l -> (q r) l")
    # query_responses_ids: [num_queries * num_responses_per_query, length]

    def loss_function(params):
        logits = state.apply_fn(state.params, query_responses_ids)

        logits_reshaped = rearrange(logits, "(q r) 1 -> q r", r=args.labels.num_labels)

        loss = optax.softmax_cross_entropy_with_integer_labels(logits_reshaped, labels).mean()

        accuracy = (logits_reshaped.argmax(axis=1) == labels).astype("float32").mean()
        return loss, accuracy

    loss, accuracy = loss_function(state.params)
    loss = jax.lax.pmean(loss, axis_name="batch")
    accuracy = jax.lax.pmean(accuracy, axis_name="batch")
    return {"loss": loss, "accuracy": accuracy}


def train(args: Args):
    if args.distributed:
        jax.distributed.initialize(
            local_device_ids=range(len(args.learner_device_ids)),
        )

    args.world_size = jax.process_count()
    local_devices = jax.local_devices()
    global_devices = jax.devices()
    learner_devices = [local_devices[d_id] for d_id in args.learner_device_ids]
    global_learner_decices = [
        global_devices[d_id + process_index * len(local_devices)]
        for process_index in range(args.world_size)
        for d_id in args.learner_device_ids
    ]
    pprint({"global_learner_decices": global_learner_decices})
    args.global_learner_decices = [str(item) for item in global_learner_decices]
    args.learner_devices = [str(item) for item in learner_devices]
    args.batch_size = int(args.local_batch_size * len(local_devices) * args.world_size)
    args.local_rank = jax.process_index()

    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SimpleNamespace()  # dummy writer
    writer.add_scalar = lambda x, y, z: None
    if args.local_rank == 0:
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
    local_seed = args.seed + args.local_rank * 100003  # Prime
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        padding_side="right",
    )
    # we use the padding token manually but do not resize the token embedding of the model
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    args.pad_token_id = tokenizer.pad_token_id
    untrained_model = FlaxAutoModelForCausalLM.from_pretrained(args.base_model)
    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key, 2)
    reward_state = create_initial_reward_state_and_models(init_key, args)

    p_train_step = jax.pmap(
        functools.partial(
            train_step,
            args=args,
        ),
        axis_name="batch",
        donate_argnums=(0,),
    )
    p_val_step = jax.pmap(
        functools.partial(
            val_step,
            args=args,
        ),
        axis_name="batch",
    )

    normalization_dataset = NormalizationDataset(
        DATASET[args.task.query_dataset],
        tokenizer,
        args.task.query_length,
        seed=local_seed,
        start_text=args.task.start_text,
        end_text=args.task.end_text,
    )
    normalization_dataloader = DataLoader(normalization_dataset, batch_size=args.rollout_batch_size)
    iter_normalization_dataloader = iter(normalization_dataloader)

    generation_config = GenerationConfig(
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

        print(
            "before normalization. "
            + f"Gain: {reward_state.params.head_params['params']['reward_gain']}"
            + f" Bias: {reward_state.params.head_params['params']['reward_bias']}"
        )

        reward_state = normalize(
            args,
            untrained_model,
            reward_state,
            iter_normalization_dataloader,
            generation_config,
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
        "vwxyzjn/lm-human-preferences",
        data_files=[args.label_dataset],
    )["train"]
    print("Num labels found in source:", len(labeled_dataset))
    print("training on", args.labels.num_train, "in batches of", args.local_batch_size)

    all_queries_responses, all_labels = prepare_left_padded_query_responses_with_labels(labeled_dataset, args)

    assert args.labels.num_train < all_queries_responses.shape[0]
    train_queries_responses = all_queries_responses[: args.labels.num_train]
    train_labels = all_labels[: args.labels.num_train]

    val_queries_responses = all_queries_responses[args.labels.num_train :]
    val_labels = all_labels[args.labels.num_train :]

    key, train_loader_key = jax.random.split(key, 2)

    train_iter = get_dataloader_iter(
        train_loader_key,
        dataset_tokens=train_queries_responses,
        dataset_labels=train_labels,
        args=args,
    )

    print("===training reward model===")

    for global_step, train_batch in enumerate(train_iter):
        train_batch = common_utils.shard(train_batch)
        reward_state, train_metrics = p_train_step(reward_state, train_batch)
        writer.add_scalar("train/lr", single_epoch_linear_schedule(global_step, args), global_step)

        # gathering replicated metric data
        train_metrics = common_utils.get_metrics([train_metrics])

        for key, value in train_metrics.items():
            writer.add_scalar(f"train/{key}", value, global_step)

        if args.print_sample_output_freq > 0 and global_step % args.print_sample_output_freq == 0:
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

            print(f"gloabl_step: {global_step} | " + f"test/accuracy {val_metrics['accuracy']}")

    reward_state = jax_utils.unreplicate(reward_state)

    if args.normalize_after:
        print("===Normalize reward model *after* training===")
        print(
            "before normalization. "
            + f"Gain: {reward_state.params.head_params['params']['reward_gain']}"
            + f" Bias: {reward_state.params.head_params['params']['reward_bias']}"
        )

        reward_state = normalize(
            args,
            untrained_model,
            reward_state,
            iter_normalization_dataloader,
            generation_config,
        )
        print(
            "after normalization. "
            + f"Gain: {reward_state.params.head_params['params']['reward_gain']}"
            + f" Bias: {reward_state.params.head_params['params']['reward_bias']}"
        )

    # save model
    if args.save_path and args.local_rank == 0:
        ckpt = {"reward_model": reward_state, "args": vars(args)}
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(ckpt)
        orbax_checkpointer.save(args.save_path, ckpt, save_args=save_args, force=True)

    if args.local_rank == 0 and args.track:
        wandb.finish()


if __name__ == "__main__":
    args = tyro.cli(Args)
    train(args)
