from __future__ import annotations

import functools
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray

from cfrx.envs import Env, InfoState, State
from cfrx.episode import Episode
from cfrx.policy import Policy, TabularPolicy
from cfrx.utils import regret_matching


class MCCFRState(NamedTuple):
    regrets: Float[Array, "*batch a"]
    probs: Float[Array, "... a"]
    avg_probs: Float[Array, "... a"]
    step: Int[Array, "..."]

    @classmethod
    def init(cls, n_states: int, n_actions: int) -> MCCFRState:
        return MCCFRState(
            regrets=jnp.zeros((n_states, n_actions)),
            probs=jnp.ones((n_states, n_actions))
            / jnp.ones((n_states, n_actions)).sum(axis=-1, keepdims=True),
            avg_probs=jnp.zeros((n_states, n_actions)) + 1e-6,
            step=jnp.array(1, dtype=int),
        )


def compute_sampled_counterfactual_action_value(
    opponent_reach_prob: Float[Array, ""],
    outcome_sampling_prob: Float[Array, ""],
    outcome_prob: Float[Array, ""],
    utility: Float[Array, ""],
) -> Float[Array, ""]:
    cf_value_a = opponent_reach_prob * outcome_prob * utility / outcome_sampling_prob
    return cf_value_a


def compute_strategy_profile(
    my_reach_prob: Float[Array, ""],
    sample_reach_prob: Float[Array, ""],
    strat_probs: Float[Array, "..."],
) -> Float[Array, "..."]:
    avg_probs = my_reach_prob / sample_reach_prob * strat_probs

    return avg_probs


def get_regrets(
    step: Episode,
    my_prob_cumprod: Float[Array, ""],
    opponent_prob_cumprod: Float[Array, ""],
    sample_prob_cumprod: Float[Array, ""],
    outcome_prob: Float[Array, ""],
    strat_prob_distrib: Float[Array, " num_actions"],
    strat_prob: Float[Array, ""],
    utility: Float[Array, ""],
    outcome_sampling_prob: Float[Array, ""],
    update_player: Int[Array, ""],
) -> tuple[Float[Array, " num_actions"], Float[Array, " num_actions"],]:
    is_current_player = step.current_player == update_player
    cf_value_a = compute_sampled_counterfactual_action_value(
        opponent_reach_prob=opponent_prob_cumprod,
        outcome_sampling_prob=outcome_sampling_prob,
        outcome_prob=outcome_prob,
        utility=utility,
    )
    cf_value = cf_value_a * strat_prob

    regrets = jnp.zeros(step.action_mask.shape[-1])
    regrets = regrets.at[step.action].set(cf_value_a)
    regrets = (regrets - cf_value) * step.action_mask

    regrets = regrets * is_current_player * step.mask * (1 - step.chance_node)

    avg_probs = compute_strategy_profile(
        my_reach_prob=my_prob_cumprod,
        sample_reach_prob=sample_prob_cumprod,
        strat_probs=strat_prob_distrib * step.action_mask,
    )
    avg_probs = avg_probs * step.mask * is_current_player * (1 - step.chance_node)
    return regrets, avg_probs


def compute_regrets_and_strategy_profile(
    episode: Episode,
    training_state: MCCFRState,
    policy: TabularPolicy,
    update_player: Int[Array, ""],
) -> tuple[InfoState[Array, "..."], Float[Array, "..."], Float[Array, "..."]]:
    episode_length = episode.current_player.shape[-1]
    utility = episode.reward[episode.mask.sum(), update_player]
    is_current_player = episode.current_player == update_player
    prob_dist_fn = functools.partial(
        policy.prob_distribution, params=training_state.probs
    )

    strat_prob_distribs = jax.vmap(prob_dist_fn)(
        info_state=episode.info_state,
        action_mask=episode.action_mask,
        use_behavior_policy=jnp.zeros(episode_length, dtype=bool),
    )

    strat_probs = jnp.where(
        episode.mask,
        strat_prob_distribs[jnp.arange(episode_length), episode.action],
        1.0,
    )
    strat_probs = jnp.where(episode.chance_node, episode.behavior_prob, strat_probs)
    my_probs = jnp.where(is_current_player, strat_probs, 1)
    opponent_probs = jnp.where(is_current_player, 1, strat_probs)
    sample_probs = jnp.where(episode.mask, episode.behavior_prob, 1)

    sample_probs_cumprod = jnp.cumprod(jnp.concatenate([jnp.ones(1), sample_probs]))
    my_probs_cumprod = jnp.cumprod(jnp.concatenate([jnp.ones(1), my_probs]))
    opponent_probs_cumprod = jnp.cumprod(opponent_probs)
    outcome_probs = jnp.cumprod(jnp.concatenate([strat_probs, jnp.ones(1)])[::-1])[::-1]

    outcome_sampling_prob = jnp.prod(sample_probs)

    _get_regrets = functools.partial(
        get_regrets,
        utility=utility,
        update_player=update_player,
        outcome_sampling_prob=outcome_sampling_prob,
    )
    regrets, avg_probs = jax.vmap(_get_regrets)(
        step=episode,
        my_prob_cumprod=my_probs_cumprod[:-1],
        opponent_prob_cumprod=opponent_probs_cumprod,
        sample_prob_cumprod=sample_probs_cumprod[:-1],
        outcome_prob=outcome_probs[1:],
        strat_prob_distrib=strat_prob_distribs,
        strat_prob=strat_probs,
    )
    regrets = regrets[:-1]
    avg_probs = avg_probs[:-1]
    episode = jax.tree_map(lambda x: x[:-1], episode)

    return episode.info_state, regrets, avg_probs


def unroll(
    init_state: State,
    random_key: PRNGKeyArray,
    training_state: MCCFRState,
    env: Env,
    policy: Policy,
    update_player: Int[Array, ""],
    n_max_steps: int,
) -> tuple[Episode, State]:
    """
    Generates a single unroll of the game.
    """

    def play_step(
        carry: tuple[State, PRNGKeyArray], unused: Any
    ) -> tuple[tuple[State, PRNGKeyArray], tuple[Episode, State]]:
        state, random_key = carry

        use_behavior_policy = state.current_player == update_player

        random_key, subkey = jax.random.split(random_key)

        action = policy.sample(
            params=training_state.probs,
            info_state=state.info_state,
            action_mask=state.legal_action_mask,
            random_key=subkey,
            use_behavior_policy=use_behavior_policy,
        )
        probs = policy.prob_distribution(
            params=training_state.probs,
            info_state=state.info_state,
            action_mask=state.legal_action_mask,
            use_behavior_policy=use_behavior_policy,
        )

        chance_probs = state.chance_prior / state.chance_prior.sum(
            axis=-1, keepdims=True
        )
        random_key, subkey = jax.random.split(random_key)
        chance_action = jax.random.choice(
            subkey,
            jnp.arange(state.chance_prior.shape[0]),
            p=chance_probs,
        )

        action = jnp.where(state.chance_node, chance_action, action)

        probs = jnp.where(state.chance_node, chance_probs[action], probs[action])

        current_player = jnp.where(state.chance_node, -1, state.current_player)
        game_step = Episode(
            info_state=state.info_state,
            action=action,
            reward=state.rewards,
            action_mask=state.legal_action_mask,
            current_player=current_player,
            behavior_prob=probs,
            chance_node=state.chance_node,
            mask=1 - state.terminated,
        )

        state = env.step(state, action)
        return (state, random_key), (game_step, state)

    (state, random_key), (episode, states) = jax.lax.scan(
        play_step, (init_state, random_key), (), length=n_max_steps
    )

    last_game_step = Episode(
        info_state=state.info_state,
        action=jnp.array(-1),
        reward=state.rewards,
        action_mask=jnp.ones(policy._n_actions, dtype=bool),
        current_player=state.current_player,
        behavior_prob=jnp.array(1.0),
        chance_node=jnp.bool_(False),
        mask=1 - state.terminated,
    )
    states = jax.tree_map(
        lambda x, y: jnp.concatenate([jnp.expand_dims(y, 0), x]),
        states,
        init_state,
    )

    episode = jax.tree_map(
        lambda x, y: jnp.concatenate([x, jnp.expand_dims(y, 0)]),
        episode,
        last_game_step,
    )
    return episode, states


def do_iteration(
    training_state: MCCFRState,
    random_key: PRNGKeyArray,
    env: Env,
    policy: TabularPolicy,
    update_player: Int[Array, ""],
) -> tuple[Float[Array, "*batch a"], Float[Array, "*batch a"], Episode]:
    """
    Do one iteration of MCCFR: traverse the game tree once and compute counterfactual
    regrets and strategy profiles.

    Args:
        training_state: The current state of the training.
        random_key: A random key.
        env: The environment.
        policy: The policy.
        update_player: The player to update.

    Returns:
        The updated regrets and average strategy profile and the episode.
    """

    # Sample one path in the game tree
    random_key, subkey = jax.random.split(random_key)
    episode, states = unroll(
        init_state=env.init(subkey),
        training_state=training_state,
        random_key=subkey,
        update_player=update_player,
        env=env,
        policy=policy,
        n_max_steps=env.max_episode_length,
    )

    # Compute counterfactual values and strategy profile
    (
        info_states,
        sampled_regrets,
        sampled_avg_probs,
    ) = compute_regrets_and_strategy_profile(
        episode=episode,
        training_state=training_state,
        policy=policy,
        update_player=update_player,
    )
    info_states_idx = jax.vmap(env.info_state_idx)(info_states)

    # Store regret and strategy profile values
    new_regrets = training_state.regrets.at[info_states_idx].add(sampled_regrets)
    new_avg_probs = training_state.avg_probs.at[info_states_idx].add(sampled_avg_probs)

    # Accumulate regrets, compute new strategy and avg strategy
    new_probs = regret_matching(new_regrets)
    new_probs /= new_probs.sum(axis=-1, keepdims=True)

    training_state = training_state._replace(
        regrets=new_regrets,
        probs=new_probs,
        avg_probs=new_avg_probs,
        step=training_state.step + 1,
    )
    return training_state
