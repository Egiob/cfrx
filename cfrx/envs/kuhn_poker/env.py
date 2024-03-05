from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pgx
import pgx.kuhn_poker
from jaxtyping import Array, Bool, Int, PRNGKeyArray
from pgx._src.dwg.kuhn_poker import CARD
from pgx._src.struct import dataclass

import cfrx
from cfrx.envs.kuhn_poker.constants import INFO_SETS
from cfrx.utils import ravel, reverse_array_lookup

CARD.append("?")
NUM_DIFFERENT_CARDS = 3
NUM_REPEAT_CARDS = 1
NUM_TOTAL_CARDS = NUM_DIFFERENT_CARDS * NUM_REPEAT_CARDS
INFO_SETS_VALUES = np.stack(list(INFO_SETS.values()))


class InfoState(NamedTuple):
    private_card: Int[Array, "..."]
    action_sequence: Int[Array, "..."]


@dataclass
class State(pgx.kuhn_poker.State):
    info_state: InfoState = InfoState(
        private_card=jnp.int8(-1), action_sequence=jnp.ones(2, dtype=jnp.int8) * -1
    )
    chance_node: Bool[Array, "..."] = jnp.bool_(False)
    chance_prior: Int[Array, "..."] = (
        jnp.ones(NUM_DIFFERENT_CARDS, dtype=int) * NUM_REPEAT_CARDS
    )


class KuhnPoker(pgx.kuhn_poker.KuhnPoker, cfrx.envs.Env):
    @classmethod
    def action_to_string(cls, action: Int[Array, ""]) -> str:
        strings = ["b", "p"]
        if action != -1:
            a = int(action) // 2
            rep = strings[a]
        else:
            rep = "?"
        return rep

    @property
    def max_episode_length(self) -> int:
        return 6

    @property
    def max_nodes(self) -> int:
        return 60

    @property
    def n_info_states(self) -> int:
        return len(INFO_SETS)

    @property
    def n_actions(self) -> int:
        return self.num_actions

    @property
    def n_players(self) -> int:
        return self.num_players

    def update_info_state(
        self, state: State, next_state: State, action: Int[Array, ""]
    ) -> InfoState:
        info_state = next_state.info_state

        private_card = jnp.where(
            next_state.chance_node, -1, next_state._cards[next_state.current_player]
        )
        current_position = (info_state.action_sequence != -1).sum()
        action_sequence = info_state.action_sequence.at[current_position].set(
            jnp.int8(action)
        )

        action_sequence = jnp.where(state.chance_node, -1, action_sequence)

        info_state = info_state._replace(
            private_card=private_card, action_sequence=action_sequence
        )
        return info_state

    def info_state_to_str(self, info_state: InfoState) -> str:
        strings = ["b", "p"]
        rep = f"{info_state.private_card}"

        for action in np.array(info_state.action_sequence):
            action = action // 2
            if action != -1:
                rep += strings[action]

        return rep

    def info_state_idx(self, info_state: InfoState) -> Int[Array, ""]:
        info_state_ravel = ravel(info_state)
        return reverse_array_lookup(info_state_ravel, jnp.asarray(INFO_SETS_VALUES))

    def _init(self, rng: PRNGKeyArray) -> State:
        env_state = super()._init(rng)

        info_state = InfoState(
            private_card=jnp.int8(-1), action_sequence=jnp.ones(2, dtype=jnp.int8) * -1
        )

        return State(
            current_player=env_state.current_player.astype(jnp.int8),
            observation=env_state.observation,
            rewards=env_state.rewards,
            terminated=env_state.terminated,
            truncated=env_state.truncated,
            _step_count=env_state._step_count,
            _last_action=env_state._last_action,
            _cards=jnp.int8([-1, -1]),
            legal_action_mask=jnp.bool_([1, 1, 1, 1]),
            _pot=env_state._pot,
            info_state=info_state,
            chance_node=jnp.bool_(True),
            chance_prior=jnp.ones(NUM_DIFFERENT_CARDS, dtype=int) * NUM_REPEAT_CARDS,
        )

    def _resolve_chance_node(
        self, state: State, action: Int[Array, ""], random_key: PRNGKeyArray
    ) -> State:
        draw_player = NUM_TOTAL_CARDS - state.chance_prior.sum()

        cards = state._cards.at[draw_player].set(action.astype(jnp.int8))
        chance_prior = state.chance_prior.at[action].add(-1)
        chance_node = (cards == -1).any()

        legal_action_mask = jnp.where(
            chance_node, state.legal_action_mask, jnp.bool_([0, 1, 0, 1])
        )

        return State(
            current_player=state.current_player.astype(jnp.int8),
            observation=state.observation,
            rewards=state.rewards,
            terminated=state.terminated,
            truncated=state.truncated,
            _step_count=state._step_count,
            _last_action=state._last_action,
            _cards=cards,
            legal_action_mask=legal_action_mask,
            _pot=state._pot,
            info_state=state.info_state,
            chance_node=chance_node,
            chance_prior=chance_prior,
        )

    def _resolve_decision_node(
        self, state: State, action: Int[Array, ""], random_key: PRNGKeyArray
    ) -> State:
        env_state = super()._step(state=state, action=action, key=random_key)

        return State(
            current_player=env_state.current_player.astype(jnp.int8),
            observation=env_state.observation,
            rewards=env_state.rewards,
            terminated=env_state.terminated,
            truncated=env_state.truncated,
            _step_count=env_state._step_count,
            _last_action=env_state._last_action,
            _cards=env_state._cards,
            legal_action_mask=env_state.legal_action_mask,
            _pot=env_state._pot,
            info_state=env_state.info_state,
            chance_node=jnp.bool_(False),
            chance_prior=env_state.chance_prior,
        )

    def _step(
        self, state: State, action: Int[Array, ""], random_key: PRNGKeyArray
    ) -> State:
        new_state = jax.lax.cond(
            state.chance_node,
            lambda: self._resolve_chance_node(
                state=state, action=action, random_key=random_key
            ),
            lambda: self._resolve_decision_node(
                state=state, action=action, random_key=random_key
            ),
        )
        info_state = self.update_info_state(
            state=state, next_state=new_state, action=action
        )
        return new_state.replace(info_state=info_state)
