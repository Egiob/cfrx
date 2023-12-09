from typing import Any, Tuple

import jax
import jax.numpy as jnp
import pgx
from jaxtyping import Array, Key

from cfrx.outcome_sampling_mccfr import Episode, MCCFRState
from cfrx.policy import Policy


def unroll(
    init_state: pgx.State,
    random_key: Key[Array, ""],
    training_state: MCCFRState,
    env: pgx.Env,
    policy: Policy,
    update_player: int,
    num_max_steps: int,
) -> Tuple[Episode, pgx.State]:
    """
    Generates a single unroll of the game.
    """

    def play_step(
        carry: Tuple[pgx.State, Key[Array, ""]], unused: Any
    ) -> Tuple[Tuple[pgx.State, Key[Array, ""]], Tuple[Episode, pgx.State]]:
        state, random_key = carry

        action_mask = state.legal_action_mask
        use_behavior_policy = state.current_player == update_player
        info_state = env.info_state_idx(state.info_state)
        random_key, subkey = jax.random.split(random_key)

        action = policy.sample(
            probs=training_state.probs,
            info_state=info_state,
            action_mask=action_mask,
            random_key=subkey,
            use_behavior_policy=use_behavior_policy,
        )
        probs = policy.probability_distribution(
            probs=training_state.probs,
            info_state=info_state,
            action_mask=action_mask,
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
            info_state_idx=info_state,
            action=action,
            reward=state.rewards,
            action_mask=action_mask,
            current_player=current_player,
            behavior_prob=probs,
            chance_node=state.chance_node,
            mask=1 - state.terminated,
        )

        state = env.step(state, action)
        return (state, random_key), (game_step, state)

    (state, random_key), (episode, states) = jax.lax.scan(
        play_step, (init_state, random_key), (), length=num_max_steps
    )

    last_game_step = Episode(
        info_state_idx=env.info_state_idx(state.info_state),
        action=jnp.array(-1),
        reward=state.rewards,
        action_mask=jnp.ones(policy._num_actions),
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
