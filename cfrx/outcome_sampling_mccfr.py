from __future__ import annotations

from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from cfrx.policy import TabularPolicy


class MCCFRState(NamedTuple):
    regrets: Float[Array, "... a"]
    probs: Float[Array, "... a"]
    average_probs: Float[Array, "... a"]
    step: Int[Array, "..."]


class Episode(NamedTuple):
    info_state_idx: Int[Array, "... a"]
    action: Float[Array, "..."]
    reward: Float[Array, "..."]
    action_mask: Bool[Array, "..."]
    current_player: Int[Array, "..."]
    behavior_prob: Float[Array, "..."]
    mask: Bool[Array, "..."]
    chance_node: Bool[Array, "..."]


def compute_sampled_counterfactual_action_value(
    opponent_reach_probability: Float[Array, ""],
    outcome_sampling_probability: Float[Array, ""],
    outcome_probability: Float[Array, ""],
    utility: Float[Array, ""],
) -> Float[Array, ""]:
    cf_value_a = (
        opponent_reach_probability
        * outcome_probability
        * utility
        / outcome_sampling_probability
    )
    return cf_value_a


def compute_strategy_profile(
    my_reach_probability: Float[Array, ""],
    sample_reach_probability: Float[Array, ""],
    strategy_probabilities: Float[Array, "..."],
) -> Float[Array, "..."]:
    average_probs = (
        my_reach_probability / sample_reach_probability * strategy_probabilities
    )

    return average_probs


def compute_regrets_and_strategy_profile(
    episode: Episode,
    training_state: MCCFRState,
    policy: TabularPolicy,
    update_player: int,
) -> Tuple[Array, Array, Array]:
    episode_length = episode.reward.shape[-1]
    utility = episode.reward[episode.mask.sum(), update_player]
    is_current_player = episode.current_player == update_player
    strategy_probability_distributions = jax.vmap(
        policy.probability_distribution, in_axes=(None,)
    )(
        training_state.probs,
        info_state=episode.info_state_idx,
        action_mask=episode.action_mask,
        use_behavior_policy=jnp.zeros_like(episode.action).astype(bool),
    )

    strategy_probabilities = jnp.where(
        episode.mask,
        strategy_probability_distributions[jnp.arange(episode_length), episode.action],
        1.0,
    )
    strategy_probabilities = jnp.where(
        episode.chance_node, episode.behavior_prob, strategy_probabilities
    )

    my_probabilities = jnp.where(is_current_player, strategy_probabilities, 1)

    opponent_probabilities = jnp.where(is_current_player, 1, strategy_probabilities)

    sample_probabilities = jnp.where(episode.mask, episode.behavior_prob, 1)

    sample_cumprod = jnp.cumprod(jnp.concatenate([jnp.ones(1), sample_probabilities]))
    outcome_sampling_probability = jnp.prod(sample_probabilities)

    my_cumprod = jnp.cumprod(jnp.concatenate([jnp.ones(1), my_probabilities]))
    opponent_cumprod = jnp.cumprod(opponent_probabilities)

    outcome_probabilities = jnp.cumprod(
        jnp.concatenate([strategy_probabilities, jnp.ones(1)])[::-1]
    )[::-1]

    def get_regret(i: Int[Array, ""]) -> Tuple[Array, Array, Array]:
        info_state_idx = episode.info_state_idx[i]
        action = episode.action[i]
        mask = episode.mask[i]
        action_mask = episode.action_mask[i]

        cf_value_a = compute_sampled_counterfactual_action_value(
            opponent_reach_probability=opponent_cumprod[i],
            outcome_sampling_probability=outcome_sampling_probability,
            outcome_probability=outcome_probabilities[i + 1],
            utility=utility,
        )
        cf_value = cf_value_a * strategy_probabilities[i]

        regrets = jnp.zeros_like(action_mask)
        regrets = regrets.at[action].set(cf_value_a)
        regrets = (regrets - cf_value) * action_mask

        regrets = regrets * is_current_player[i] * mask * (1 - episode.chance_node[i])

        average_probs = compute_strategy_profile(
            my_reach_probability=my_cumprod[i],
            sample_reach_probability=sample_cumprod[i],
            strategy_probabilities=strategy_probability_distributions[i] * action_mask,
        )

        average_probs = (
            average_probs * mask * is_current_player[i] * (1 - episode.chance_node[i])
        )

        return info_state_idx, regrets, average_probs

    info_state_idx, regrets, average_probs = jax.vmap(get_regret)(
        jnp.arange(episode_length - 1),
    )

    return info_state_idx, regrets, average_probs
