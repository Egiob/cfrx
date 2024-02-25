from abc import ABC, abstractmethod
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from cfrx.envs.base import InfoState


class Policy(ABC):
    _n_actions: int

    @abstractmethod
    def prob_distribution(
        self,
        params: Float[Array, "... a"],
        info_state: InfoState[Array, "..."],
        action_mask: Bool[Array, "... a"],
        use_behavior_policy: Bool[Array, "..."],
    ) -> Array:
        pass

    @abstractmethod
    def sample(
        self,
        params: Float[Array, "... a"],
        info_state: InfoState[Array, "..."],
        action_mask: Bool[Array, "... a"],
        random_key: PRNGKeyArray,
        use_behavior_policy: Bool[Array, "..."],
    ) -> Array:
        pass


class TabularPolicy(Policy):
    def __init__(
        self,
        n_actions: int,
        info_state_idx_fn: Callable[[InfoState], Int[Array, ""]],
        exploration_factor: float = 0.6,
    ):
        self._n_actions = n_actions
        self._exploration_factor = exploration_factor
        self._info_state_idx_fn = info_state_idx_fn

    def prob_distribution(
        self,
        params: Float[Array, "... a"],
        info_state: InfoState[Array, "..."],
        action_mask: Bool[Array, "... a"],
        use_behavior_policy: Bool[Array, "..."],
    ) -> Array:
        info_state_idx = self._info_state_idx_fn(info_state)
        probs = params[info_state_idx]

        behavior_probabilities = (
            probs * (1 - self._exploration_factor)
            + self._exploration_factor * jnp.ones_like(probs) / self._n_actions
        )

        probs = jnp.where(use_behavior_policy, behavior_probabilities, probs)
        probs = probs * action_mask
        probs /= probs.sum(axis=-1, keepdims=True)
        return probs

    def sample(
        self,
        params: Float[Array, "... a"],
        info_state: InfoState[Array, "..."],
        action_mask: Bool[Array, "... a"],
        random_key: PRNGKeyArray,
        use_behavior_policy: Bool[Array, "..."],
    ) -> Array:
        info_state_idx = self._info_state_idx_fn(info_state)
        probs = params[info_state_idx]

        behavior_probabilities = (
            probs * (1 - self._exploration_factor)
            + self._exploration_factor * jnp.ones_like(probs) / self._n_actions
        )

        probs = jnp.where(use_behavior_policy, behavior_probabilities, probs)

        probs = probs * action_mask
        probs /= probs.sum(axis=-1, keepdims=True)
        action = jax.random.choice(random_key, jnp.arange(self._n_actions), p=probs)
        return action
