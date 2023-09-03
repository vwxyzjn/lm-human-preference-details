import functools
import os
import time
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace
from typing import List, Optional

import einops
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax
import tyro
from flax import jax_utils
from flax.training import common_utils, orbax_utils
from flax.training.train_state import TrainState
from optax import ScaleByAdamState, update_moment, update_moment_per_elem_norm
from optax._src import base, combine, numerics, utils
from optax._src.alias import _scale_by_learning_rate
from rich.console import Console
from rich.pretty import pprint
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, FlaxAutoModelForCausalLM, GenerationConfig

from lm_human_preference_details.data import DATASET


@dataclass
class AdaptiveKLParams:
    target: float = 6.0
    horizon: int = 10000  # in episodes


@dataclass
class RewardHParams:
    kl_coef: float = 0.15
    adaptive_kl: Optional[AdaptiveKLParams] = field(default_factory=AdaptiveKLParams)
    trained_model: Optional[str] = "models/"
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
    query_length: int = 64
    query_dataset: str = "books"
    query_prefix: str = ""
    query_suffix: str = ""
    start_text: Optional[str] = None
    end_text: Optional[str] = None

    # Response params
    response_length: int = 24

    # Truncate response after the first occurrence of this token at or after index after when sampling.
    truncate_token: int = 13
    truncate_after: int = 16
    penalty_reward_value: int = -1

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
    print_sample_output_freq: int = 0
    """How often to print sample output"""
    save_path: str = "models/policy/"
    """Where to save the model"""
    use_tensorflow_adam: bool = True
    """Whether to use tensorflow-style Adam optimizer instead of PyTorch's"""
    task: TaskHParams = field(default_factory=TaskHParams)
    rewards: RewardHParams = field(default_factory=RewardHParams)
    ppo: PpoHParams = field(default_factory=PpoHParams)

    # distributed settings
    local_rank: int = 0
    """the rank of this process"""
    learner_device_ids: List[int] = field(default_factory=lambda: [0])
    "the device ids that script will use"
    learner_devices: tyro.conf.Suppress[int] = None  # real type is `List[str]`
    """the devices that script will use"""
    global_learner_decices: tyro.conf.Suppress[int] = None  # real type is `List[str]`
    """the total devices (across all nodes and machines) that script will use"""


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


class AdaptiveKLController:
    def __init__(self, init_kl_coef: float, hparams: AdaptiveKLParams):
        self.value = init_kl_coef
        self.hparams = hparams

    def update(self, current, n_steps):
        target = self.hparams.target
        proportional_error = np.clip(current / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.hparams.horizon
        self.value *= mult


def whiten(values, shift_mean=True):
    # `unbiased=False` matches TF `tf.nn.moments`'s setting
    mean, var = jnp.mean(values), jnp.var(values)
    whitened = (values - mean) * jax.lax.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


class ScalarHead(nn.Module):
    head_input_size: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(
            1,
            kernel_init=nn.initializers.normal(stddev=0),
            bias_init=nn.initializers.zeros_init(),
        )(x)
        return x


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


@flax.struct.dataclass
class LMBackboneWithScalarHeadParams:
    """Parameters for the language model backbone and a scalar head."""

    lm_backbone_params: flax.core.FrozenDict
    head_params: flax.core.FrozenDict


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


def right_padding_to_left_padding(tokens, pad_id):
    # Convert from right padding to left padding.
    return np.array([[pad_id] * (row == pad_id).sum() + [x for x in row if x != pad_id] for row in tokens])


def ceil_div(a, b):
    return (a - 1) // b + 1


def exact_div(a, b):
    q = a // b
    if a != q * b:
        raise ValueError(f"Inexact division: {a} / {b} = {a / b}")
    return q


def prepare_reward_forward(args, tokenizer):
    """Prepare the forward pass of the reward model and parameters."""

    lm_backbone = FlaxAutoModelForCausalLM.from_pretrained(args.base_model)
    scalar_head = RewardHead(head_input_size=lm_backbone.config.hidden_size)

    def reward_forward(
        params: LMBackboneWithScalarHeadParams,
        query_responses_ids: jnp.ndarray,
    ):
        """Get reward for each queiry--response pair."""
        assert query_responses_ids.ndim == 2

        # mask out padding tokens
        attention_mask = query_responses_ids != tokenizer.pad_token_id
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
        # reward_latents: [batch_size, length, hidden_size]

        last_reward_latents = reward_latents[:, -1, :]
        # last_reward_latents: [batch_size, hidden_size]

        reward = scalar_head.apply(variables=params.head_params, x=last_reward_latents)
        # reward: [batch_size, 1]
        return reward

    if args.rewards.trained_model:
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        reward_state_params = orbax_checkpointer.restore(args.rewards.trained_model)["reward_model"]["params"]
        reward_params = LMBackboneWithScalarHeadParams(
            lm_backbone_params=flax.core.FrozenDict({"params": reward_state_params["lm_backbone_params"]["params"]}),
            head_params=flax.core.FrozenDict({"params": reward_state_params["head_params"]["params"]}),
        )
        pprint(f"Loaded pretrained reward model from {args.rewards.trained_model}")
    else:
        key = jax.random.PRNGKey(args.seed)
        key, init_key = jax.random.split(key, 2)
        reward_params = LMBackboneWithScalarHeadParams(
            lm_backbone_params=flax.core.FrozenDict({"params": lm_backbone.params}),
            head_params=flax.core.FrozenDict(
                scalar_head.init(
                    init_key,
                    jnp.ones(lm_backbone.config.hidden_size)[None, None, :],
                )
            ),
        )

    return functools.partial(reward_forward, params=reward_params)


def prepare_policy_forward_and_policy_generate(args, tokenizer):
    """Prepare the forward pass of the policy model and parameters."""

    lm_backbone = FlaxAutoModelForCausalLM.from_pretrained(args.base_model)
    # disable `pad_token_id` and `eos_token_id` because we just want to
    # generate tokens without truncation / padding
    lm_backbone.generation_config.eos_token_id = None
    lm_backbone.generation_config.pad_token_id = None
    scalar_head = ScalarHead(head_input_size=lm_backbone.config.hidden_size)

    generation_config = GenerationConfig(
        max_new_tokens=args.task.response_length,
        min_new_tokens=args.task.response_length,
        temperature=args.task.temperature,
        top_k=0.0,
        top_p=1.0,
        do_sample=True,
        eos_token_id=None,
        pad_token_id=tokenizer.pad_token_id,
    )

    def policy_forward(
        params: LMBackboneWithScalarHeadParams,
        input_ids: jnp.ndarray,
    ):
        """Get reward for input_ids."""
        assert input_ids.ndim == 2
        # shape: [batch_size, length]

        # mask out padding tokens
        attention_mask = input_ids != tokenizer.pad_token_id
        input_ids = jnp.where(attention_mask, input_ids, 0)

        # assign position ids
        position_ids = attention_mask.cumsum(1) - attention_mask

        lm_backbone_out = lm_backbone.module.apply(
            variables=params.lm_backbone_params,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
        )

        value_latents = lm_backbone_out.hidden_states[-1]
        # shape: [batch_size, length, hidden_size]

        values = scalar_head.apply(variables=params.head_params, x=value_latents)
        # shape: [batch_size, length, 1]
        return lm_backbone_out, values

    def policy_generate(
        params: LMBackboneWithScalarHeadParams,
        input_ids: jnp.ndarray,
    ):
        attention_mask = input_ids != tokenizer.pad_token_id
        input_ids = jnp.where(attention_mask, input_ids, 0)
        output = lm_backbone.generate(
            params=params["params"],
            input_ids=input_ids,
            generation_config=generation_config,
            attention_mask=attention_mask.astype("i4"),
            return_dict_in_generate=True,
        )
        context_length = input_ids.shape[1]
        return jnp.concatenate((input_ids, output.sequences[:, context_length:]), axis=1)

    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key, 2)
    policy_params = LMBackboneWithScalarHeadParams(
        lm_backbone_params=flax.core.FrozenDict({"params": lm_backbone.params}),
        head_params=flax.core.FrozenDict(
            scalar_head.init(
                init_key,
                jnp.ones(lm_backbone.config.hidden_size)[None, None, :],
            )
        ),
    )

    return policy_forward, policy_generate, policy_params


@flax.struct.dataclass
class RolloutStatistics:
    returns: jnp.array
    values: jnp.array
    advantage: jnp.array
    responses: jnp.array
    query_responses: jnp.array
    logprobs: jnp.array


@flax.struct.dataclass
class RLStatistics:
    approxkl: jnp.array
    entropy: jnp.array
    pg_loss: jnp.array
    pg_clipfrac: jnp.array
    vf_losses1: jnp.array
    vf_loss: jnp.array
    vf_clipfrac: jnp.array
    ratio: jnp.array
    loss: jnp.array


def train_step(policy_state, mb_stats, args):
    def loss(params):
        # mb_stats.query_responses: [local_micro_batch_size, query_length + response_length]
        output, vpred_temp = policy_state.apply_fn(params, mb_stats.query_responses)
        # vpred_temp: [local_micro_batch_size, query_length + response_length, 1]
        vpred = jnp.squeeze(vpred_temp[:, args.task.query_length - 1 : -1, :], axis=-1)
        # vpred: [local_micro_batch_size, response_length]
        vpredclipped = jnp.clip(
            vpred,
            mb_stats.values - args.ppo.cliprange_value,
            mb_stats.values + args.ppo.cliprange_value,
        )
        vf_losses1 = jnp.square(vpred - mb_stats.returns)
        vf_losses2 = jnp.square(vpredclipped - mb_stats.returns)
        vf_loss = 0.5 * jnp.maximum(vf_losses1, vf_losses2).mean()
        vf_clipfrac = (vf_losses2 > vf_losses1).astype(jnp.float32).mean()

        logits = output.logits[:, args.task.query_length - 1 : -1, :]
        logits /= args.task.temperature
        new_logprobs = -optax.softmax_cross_entropy_with_integer_labels(logits, mb_stats.responses)

        logprobs_diff = new_logprobs - mb_stats.logprobs
        ratio = jnp.exp(logprobs_diff)
        pg_losses = -mb_stats.advantage * ratio
        pg_losses2 = -mb_stats.advantage * jnp.clip(ratio, 1.0 - args.ppo.cliprange, 1.0 + args.ppo.cliprange)

        pg_loss = jnp.maximum(pg_losses, pg_losses2).mean()
        pg_clipfrac = (pg_losses2 > pg_losses).astype(jnp.float32).mean()

        pd = jax.nn.softmax(logits, axis=-1)
        entropy = jax.nn.logsumexp(logits, axis=-1) - jnp.sum(pd * logits, axis=-1)

        approxkl = 0.5 * ((logprobs_diff) ** 2).mean()
        loss = pg_loss + args.ppo.vf_coef * vf_loss

        rl_stats = RLStatistics(
            vf_loss=vf_loss,
            vf_clipfrac=vf_clipfrac,
            pg_loss=pg_loss,
            pg_clipfrac=pg_clipfrac,
            approxkl=approxkl,
            loss=loss,
            entropy=entropy.mean(),
            ratio=ratio.mean(),
            vf_losses1=vf_losses1.mean(),
        )
        rl_stats = jax.lax.pmean(rl_stats, "batch")
        return loss, rl_stats

    grad_fn = jax.value_and_grad(loss, has_aux=True)
    (loss, rl_stats), grads = grad_fn(policy_state.params)
    grads = jax.lax.pmean(grads, "batch")
    policy_state = policy_state.apply_gradients(grads=grads)
    return policy_state, rl_stats


def linear_schedule(optimizer_step, args):
    """anneal learning rate linearly to reach 0 after one epoch."""
    update = 1 + optimizer_step // (args.ppo.noptepochs * args.ppo.nminibatches * args.ppo.gradient_accumulation_steps)
    frac = 1.0 - (update - 1.0) / args.ppo.num_updates
    lrnow = frac * args.ppo.lr
    return lrnow


def train(args: Args):
    local_devices = jax.local_devices()
    global_devices = jax.devices()
    args.ppo.world_size = jax.process_count()
    learner_devices = [local_devices[d_id] for d_id in args.learner_device_ids]
    global_learner_decices = [
        global_devices[d_id + process_index * len(local_devices)]
        for process_index in range(args.ppo.world_size)
        for d_id in args.learner_device_ids
    ]
    pprint({"global_learner_decices": global_learner_decices})
    args.global_learner_decices = [str(item) for item in global_learner_decices]
    args.learner_devices = [str(item) for item in learner_devices]
    args.local_rank = jax.process_index()
    args.ppo.batch_size = int(args.ppo.local_batch_size * len(args.learner_devices) * args.ppo.world_size)
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

    console = Console(force_terminal=True)
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SimpleNamespace()  # dummy writer
    writer.add_scalar = lambda x, y, z: None
    writer.add_histogram = lambda x, y, z: None

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
    reward_forward = prepare_reward_forward(args, tokenizer)
    policy_forward, policy_generate, policy_params = prepare_policy_forward_and_policy_generate(args, tokenizer)
    _, _, ref_policy_params = prepare_policy_forward_and_policy_generate(args, tokenizer)

    p_whiten_no_shift_mean = jax.pmap(functools.partial(whiten, shift_mean=False))
    p_whiten_shift_mean = jax.pmap(functools.partial(whiten, shift_mean=True))
    p_reward_forward = jax.pmap(reward_forward)
    p_policy_generate = jax.pmap(policy_generate)
    p_policy_forward = jax.pmap(policy_forward)
    p_train_step = jax.pmap(
        functools.partial(train_step, args=args),
        axis_name="batch",
        donate_argnums=(0,),
    )

    if args.use_tensorflow_adam:
        adam = adam_tf_style
    else:
        adam = optax.adam

    optimizer = adam(
        learning_rate=functools.partial(linear_schedule, args=args),
        eps=args.ppo.eps,
    )

    optimizer = optax.MultiSteps(optimizer, args.ppo.gradient_accumulation_steps)

    policy_state = TrainState.create(apply_fn=policy_forward, params=policy_params, tx=optimizer)
    policy_state = jax_utils.replicate(policy_state)
    ref_policy_params = jax_utils.replicate(ref_policy_params)

    del policy_params

    dataset = MyDataset(
        DATASET[args.task.query_dataset],
        tokenizer,
        args.task.query_length,
        seed=local_seed,
        start_text=args.task.start_text,
        end_text=args.task.end_text,
    )
    dataloader = DataLoader(dataset, batch_size=args.ppo.batch_size)
    iter_dataloader = iter(dataloader)
    kl_ctl = AdaptiveKLController(args.rewards.kl_coef, hparams=args.rewards.adaptive_kl)

    print("===training policy===")
    global_step = 0
    approxkls_stats = np.zeros(
        (
            args.ppo.noptepochs,
            args.ppo.nminibatches,
            args.ppo.gradient_accumulation_steps,
        ),
    )
    pg_clipfracs_stats = np.zeros(
        (
            args.ppo.noptepochs,
            args.ppo.nminibatches,
            args.ppo.gradient_accumulation_steps,
        ),
    )
    pg_losses_stats = np.zeros(
        (
            args.ppo.noptepochs,
            args.ppo.nminibatches,
            args.ppo.gradient_accumulation_steps,
        ),
    )
    vf_losses_stats = np.zeros(
        (
            args.ppo.noptepochs,
            args.ppo.nminibatches,
            args.ppo.gradient_accumulation_steps,
        ),
    )
    vf_clipfrac_stats = np.zeros(
        (
            args.ppo.noptepochs,
            args.ppo.nminibatches,
            args.ppo.gradient_accumulation_steps,
        ),
    )
    entropies_stats = np.zeros(
        (
            args.ppo.noptepochs,
            args.ppo.nminibatches,
            args.ppo.gradient_accumulation_steps,
        ),
    )
    for update in range(1, args.ppo.num_updates + 1):
        global_step += 1 * args.ppo.batch_size
        data = next(iter_dataloader)
        queries = right_padding_to_left_padding(data["input_ids"], tokenizer.pad_token_id)
        queries = common_utils.shard(queries)
        # queries: [num_device, local_batch_size, query_length]

        query_responses = p_policy_generate(
            params=policy_state.params.lm_backbone_params,
            input_ids=queries,
        )
        # query_responses: [num_device, local_batch_size, query_length + response_length]
        responses = query_responses[..., args.task.query_length :]

        output, full_values = p_policy_forward(policy_state.params, query_responses)
        values = full_values[:, :, args.task.query_length - 1 : -1].squeeze(-1)
        # values: [num_device, local_batch_size, response_length]
        logits = output.logits[:, :, args.task.query_length - 1 : -1, :]
        # logits: [num_device, local_batch_size, response_length, vocab_size]
        logits /= args.task.temperature
        all_logprobs = jax.nn.log_softmax(logits, axis=-1)
        # all_logprobs: [num_device, local_batch_size, response_length, vocab_size]
        logprobs = jnp.take_along_axis(all_logprobs, responses[..., None], -1).squeeze(-1)
        # logprobs: [num_device, local_batch_size, response_length]
        del output, logits, all_logprobs

        ref_output, _ = p_policy_forward(ref_policy_params, query_responses)
        ref_logits = ref_output.logits[:, :, args.task.query_length - 1 : -1, :]
        # ref_logits: [num_device, local_batch_size, response_length, vocab_size]
        ref_logits /= args.task.temperature
        ref_all_logprobs = jax.nn.log_softmax(ref_logits, axis=-1)
        ref_logprobs = jnp.take_along_axis(ref_all_logprobs, responses[..., None], -1).squeeze(-1)
        # ref_logprobs: [num_device, local_batch_size, response_length]
        del ref_output, ref_logits, ref_all_logprobs

        # **Response Processing**
        # 1. truncate at the first occurrence of `truncate_token` that appears at or after
        # position truncate_after in the responses
        # https://github.com/openai/lm-human-preferences/blob/cbfd210bb8b08f6bc5c26878c10984b90f516c66/lm_human_preferences/train_policy.py#L378
        truncate_token_mask = responses == args.task.truncate_token
        # truncate_token_mask: [num_device, local_batch_size, response_length]
        truncate_after_or_token_mask = jnp.concatenate(
            [
                jnp.zeros_like(truncate_token_mask)[:, :, : args.task.truncate_after],
                truncate_token_mask[:, :, args.task.truncate_after :],
            ],
            axis=-1,
        )
        truncate_mask = (jnp.cumsum(truncate_after_or_token_mask, axis=-1) - truncate_after_or_token_mask).astype("bool")
        postprocessed_responses = jnp.where(
            truncate_mask,
            jnp.full_like(responses, tokenizer.pad_token_id),
            responses,
        )
        # postprocessed_responses: [num_device, local_batch_size, response_length]
        del truncate_token_mask, truncate_after_or_token_mask, truncate_mask

        # 2. run reward model on the truncated responses
        postprocessed_query_responses = np.concatenate((queries, postprocessed_responses), axis=-1)
        # postprocessed_query_responses: [num_device, local_batch_size, query_length + response_length]
        postprocessed_query_responses = einops.rearrange(
            right_padding_to_left_padding(
                einops.rearrange(postprocessed_query_responses, "d b l -> (d b) l"), tokenizer.pad_token_id
            ),
            "(d b) l -> d b l",
            d=len(args.learner_devices),
        )
        scores = p_reward_forward(query_responses_ids=postprocessed_query_responses).squeeze(-1)
        # scores: [num_device, local_batch_size]

        # 3. filter response. Ensure that the sample contains truncate_token
        # responses not passing that filter will receive a low (fixed) score
        # only query humans on responses that pass that filter
        matches_token = postprocessed_responses[..., args.task.truncate_after :] == args.task.truncate_token
        # matches_token: [num_device, local_batch_size, response_length - args.task.truncate_after]

        filter_mask = jnp.any(matches_token, axis=-1)
        # filter_mask: [num_device, local_batch_size]

        scores = jnp.where(
            filter_mask,
            scores,
            jnp.full_like(scores, args.task.penalty_reward_value),
        )
        # scores: [num_device, local_batch_size]
        del matches_token, filter_mask

        # 4. compute rewards
        kl = logprobs - ref_logprobs
        # kl: [num_device, local_batch_size, response_length]
        non_score_reward = -kl_ctl.value * kl
        rewards = non_score_reward
        rewards = rewards.at[..., -1].add(scores)
        # rewards: [num_device, local_batch_size, response_length]

        # 5. whiten rewards
        if args.ppo.whiten_rewards:
            rewards = p_whiten_no_shift_mean(rewards)
        try:
            sample_kl = kl[0][0].sum().item()
            # postprocessed_responses = postprocessed_query_responses[..., args.task.query_length :]
            console.print(
                f"[green][bold]{'Query'}:[/]\n"
                + f"[green]{ tokenizer.decode(queries[0][0], skip_special_tokens=True)}[/]\n\n"
                + f"[blue][bold]{'Raw response'}:[/]\n"
                + f"[blue]{tokenizer.decode(responses[0][0], skip_special_tokens=True)}[/]\n\n"
                + f"[yellow][bold]{'Processed response'}:[/]\n"
                + f"[yellow]{tokenizer.decode(postprocessed_responses[0][0], skip_special_tokens=True)}[/]\n\n"
                + f"[red]score: {scores[0][0]}, kl: {kl[0][0].sum().item()}, total reward: {scores[0][0] - kl_ctl.value * sample_kl} [/]"
            )
        except Exception as e:
            print(e)
        del postprocessed_query_responses

        # 6. compute advantages and returns
        lastgaelam = 0
        advantages_reversed = []
        gen_length = args.task.response_length

        for t in reversed(range(gen_length)):
            nextvalues = values[..., t + 1] if t < gen_length - 1 else 0.0
            delta = rewards[..., t] + args.ppo.gamma * nextvalues - values[..., t]
            lastgaelam = delta + args.ppo.gamma * args.ppo.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
            # advantages_reversed is a list of 2D arrays
            # Each array has the shape [num_devices, local_batch_size]

        advantages = jnp.stack(advantages_reversed[::-1], axis=-1)
        # advantages: [num_device, local_batch_size, response_length]
        returns = advantages + values
        # returns: [num_device, local_batch_size, response_length]
        advantages = p_whiten_shift_mean(advantages)

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
                    micro_batch_end = micro_batch_start + args.ppo.local_micro_batch_size
                    micro_batch_inds = mini_batch_inds[micro_batch_start:micro_batch_end]
                    mb_returns = returns[:, micro_batch_inds, :]
                    mb_advantage = advantages[:, micro_batch_inds, :]
                    mb_values = values[:, micro_batch_inds, :]
                    mb_responses = responses[:, micro_batch_inds, :]
                    mb_query_responses = query_responses[:, micro_batch_inds, :]
                    mb_logprobs = logprobs[:, micro_batch_inds, :]
                    mb_stats = RolloutStatistics(
                        returns=mb_returns,
                        values=mb_values,
                        advantage=mb_advantage,
                        responses=mb_responses,
                        query_responses=mb_query_responses,
                        logprobs=mb_logprobs,
                    )

                    # before training step
                    policy_state, rl_stats = p_train_step(policy_state, mb_stats)
                    rl_stats = common_utils.get_metrics([rl_stats])

                    approxkls_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = rl_stats.approxkl
                    pg_clipfracs_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = rl_stats.pg_clipfrac
                    pg_losses_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = rl_stats.pg_loss
                    vf_losses_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = rl_stats.vf_loss
                    vf_clipfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = rl_stats.vf_clipfrac
                    entropies_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = rl_stats.entropy

                    gradient_accumulation_idx += 1
                minibatch_idx += 1
                if args.local_rank == 0:
                    console.print(
                        f"ppo_epoch_idx",
                        ppo_epoch_idx,
                        "approxkl",
                        rl_stats.approxkl.item(),
                        "pg_loss",
                        rl_stats.pg_loss.item(),
                        "pg_clipfrac",
                        rl_stats.pg_clipfrac.item(),
                        "ratio",
                        rl_stats.ratio.item(),
                    )

        # Rollout metrics
        mean_kl = kl.sum(-1).mean()
        mean_entropy = (-logprobs).sum(-1).mean()
        mean_non_score_reward = non_score_reward.sum(-1).mean()
        writer.add_scalar("objective/kl_coef", np.array(kl_ctl.value), update)
        writer.add_scalar("objective/kl", mean_kl.item(), update)
        writer.add_scalar("objective/entropy", mean_entropy.item(), update)
        writer.add_scalar("objective/non_score_reward", mean_non_score_reward.item(), update)
        writer.add_scalar("objective/score_total", mean_non_score_reward.item() + scores.mean().item(), update)
        writer.add_scalar("objective/scores", scores.mean().item(), update)
        writer.add_scalar("ppo/returns/mean", return_mean.item(), update)
        writer.add_scalar("ppo/returns/var", return_var.item(), update)
        writer.add_scalar("ppo/val/mean", value_mean.item(), update)
        writer.add_scalar("ppo/val/var", value_var.item(), update)
        writer.add_scalar("ppo/val/advantage", advantages.mean().item(), update)
        writer.add_scalar("ppo/val/num_eos_tokens", (responses == tokenizer.eos_token_id).sum().item(), update)

        # RL metrics aggregated at the batch level
        writer.add_scalar("ppo/policy/approxkl_avg", approxkls_stats.mean().item(), update)
        writer.add_scalar(
            "ppo/val/clipfrac_avg",
            # TODO: change the name to pg_clipfrac_avg and distinguish it from vf_clipfrac_avg?
            pg_clipfracs_stats.mean().item(),
            update,
        )
        writer.add_scalar(
            "ppo/policy/entropy_avg",
            entropies_stats.mean().item(),
            update,
        )
        writer.add_scalar(
            "ppo/loss/policy_avg",
            pg_losses_stats.mean().item(),
            update,
        )
        writer.add_scalar(
            "ppo/loss/value_avg",
            vf_losses_stats.mean().item(),
            update,
        )

        # RL metrics directly from the microbatch level.
        # TODO: Convert them to batch-level aggregations for consistency?
        # (some metrics are repetitive with the batch level ones)
        writer.add_scalar("ppo/loss/policy", rl_stats.pg_loss.item(), update)
        writer.add_scalar("ppo/loss/value", rl_stats.vf_loss.item(), update)
        writer.add_scalar("ppo/loss/total", rl_stats.loss.item(), update)
        writer.add_scalar(
            "ppo/policy/clipfrac",
            rl_stats.pg_clipfrac.item(),
            update,
        )
        writer.add_scalar(
            "ppo/policy/approxkl",
            rl_stats.approxkl.item(),
            update,
        )
        writer.add_scalar(
            "ppo/policy/entropy",
            rl_stats.entropy.item(),
            update,
        )
        writer.add_scalar(
            "ppo/val/error",
            rl_stats.vf_losses1.item(),
            update,
        )

        # Logging learning rate and learning progress
        lrnow = linear_schedule(policy_state.step - 1, args)
        lrnow = common_utils.get_metrics([lrnow])
        writer.add_scalar("ppo/lr", lrnow.item(), update)
        writer.add_scalar("ppo/episode", global_step, update)
        kl_ctl.update(mean_kl, args.ppo.batch_size)

    # save model
    if args.local_rank == 0:
        if args.save_path:
            ckpt = {"policy_model": policy_state, "args": vars(args)}
            orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
            save_args = orbax_utils.save_args_from_target(ckpt)
            orbax_checkpointer.save(args.save_path, ckpt, save_args=save_args, force=True)

        if args.local_rank == 0 and args.track:
            wandb.finish()


if __name__ == "__main__":
    args = tyro.cli(Args)
    train(args)
