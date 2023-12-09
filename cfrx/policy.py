from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, Key


class Policy(ABC):
    _num_actions: int

    @abstractmethod
    def probability_distribution(
        self,
        probs: Float[Array, "... a"],
        info_state: Int[Array, "..."],
        action_mask: Bool[Array, "... a"],
        use_behavior_policy: Bool[Array, "..."],
    ) -> Array:
        pass

    @abstractmethod
    def sample(
        self,
        probs: Float[Array, "... a"],
        info_state: Int[Array, "..."],
        action_mask: Bool[Array, "... a"],
        random_key: Key[Array, ""],
        use_behavior_policy: Bool[Array, "..."],
    ) -> Array:
        pass


class TabularPolicy(Policy):
    def __init__(
        self, num_observations: int, num_actions: int, exploration_factor: float = 0.2
    ):
        self._num_observations = num_observations
        self._num_actions = num_actions
        self._exploration_factor = exploration_factor

    def probability_distribution(
        self,
        probs: Float[Array, "... a"],
        info_state: Int[Array, "..."],
        action_mask: Bool[Array, "... a"],
        use_behavior_policy: Bool[Array, "..."],
    ) -> Array:
        probs = probs[info_state]

        behavior_probabilities = (
            probs * (1 - self._exploration_factor)
            + self._exploration_factor * jnp.ones_like(probs) / self._num_actions
        )

        probs = jnp.where(use_behavior_policy, behavior_probabilities, probs)
        probs = probs * action_mask
        probs /= probs.sum(axis=-1, keepdims=True)
        return probs

    def sample(
        self,
        probs: Float[Array, "... a"],
        info_state: Int[Array, "..."],
        action_mask: Bool[Array, "... a"],
        random_key: Key[Array, ""],
        use_behavior_policy: Bool[Array, "..."],
    ) -> Array:
        probs = probs[info_state]

        behavior_probabilities = (
            probs * (1 - self._exploration_factor)
            + self._exploration_factor * jnp.ones_like(probs) / self._num_actions
        )

        probs = jnp.where(use_behavior_policy, behavior_probabilities, probs)

        probs = probs * action_mask
        probs /= probs.sum(axis=-1, keepdims=True)
        action = jax.random.choice(random_key, jnp.arange(self._num_actions), p=probs)
        return action
