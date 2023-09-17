import jax
from jax import config

config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np


def test_compute_gae():
    def compute_gae_for_loop(
        rewards: jnp.ndarray,
        values: jnp.ndarray,
        ppo_gamma: float = 1.0,
        ppo_lam: float = 0.95,
    ):
        assert rewards.shape == values.shape
        gen_length = values.shape[1]
        advantages = 0
        advantages_reversed = []
        for t in reversed(range(gen_length)):
            nextvalues = values[:, t + 1] if t < gen_length - 1 else 0.0
            delta = rewards[:, t] + ppo_gamma * nextvalues - values[:, t]
            advantages = delta + ppo_gamma * ppo_lam * advantages
            advantages_reversed.append(advantages)
        advantages_forloop = jnp.stack(advantages_reversed[::-1], axis=1)
        return advantages_forloop

    def compute_gae_scan(
        rewards: jnp.ndarray,
        values: jnp.ndarray,
        ppo_gamma: float = 1.0,
        ppo_lam: float = 0.95,
    ):
        assert rewards.shape == values.shape

        def compute_gae_once(carry, inp):
            advantages = carry
            nextdone, nextvalues, curvalues, reward = inp
            nextnonterminal = 1.0 - nextdone

            delta = reward + ppo_gamma * nextvalues * nextnonterminal - curvalues
            advantages = delta + ppo_gamma * ppo_lam * nextnonterminal * advantages
            return advantages, advantages

        extended_values = jnp.concatenate((values, jnp.zeros((values.shape[0], 1))), axis=1)
        dones = jnp.zeros_like(rewards)
        dones = dones.at[:, -1].set(1.0)

        advantages_scan = jnp.zeros((values.shape[0],))
        _, advantages_scan = jax.lax.scan(
            compute_gae_once,
            advantages_scan,
            (dones.T, extended_values[:, 1:].T, extended_values[:, :-1].T, rewards.T),
            reverse=True,
        )

        advantages_scan = advantages_scan.T
        return advantages_scan

    key = jax.random.PRNGKey(42)
    rewards = jax.random.uniform(key, (20, 40), dtype=jnp.float64)

    key, subkey = jax.random.split(key, 2)
    values = jax.random.uniform(key, (20, 40), dtype=jnp.float64)

    advantages_forloop = compute_gae_for_loop(rewards, values)

    advantages_scan = compute_gae_scan(rewards, values)

    np.testing.assert_array_almost_equal(advantages_forloop, advantages_scan)
