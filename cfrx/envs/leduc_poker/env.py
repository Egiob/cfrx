from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pgx
import pgx.leduc_holdem
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray, Shaped
from pgx._src.dwg.leduc_holdem import CARD
from pgx._src.struct import dataclass

import cfrx.envs
from cfrx.envs.leduc_poker.constants import INFO_SETS
from cfrx.utils import ravel, reverse_array_lookup

CARD.append("?")
INFO_SETS_VALUES = np.stack(list(INFO_SETS.values()))
NUM_DIFFERENT_CARDS = 3
NUM_REPEAT_CARDS = 2
NUM_TOTAL_CARDS = NUM_DIFFERENT_CARDS * NUM_REPEAT_CARDS


class InfoState(NamedTuple):
    private_card: Int[Array, ""]
    public_card: Int[Array, ""]
    action_sequence: Int[Array, "..."]


@dataclass
class State(pgx.leduc_holdem.State):
    info_state: InfoState = InfoState(
        private_card=jnp.int8(-1),
        public_card=jnp.int8(-1),
        action_sequence=jnp.ones((2, 4), dtype=jnp.int8) * -1,
    )
    chance_node: Bool[Array, ""] = jnp.bool_(False)
    chance_prior: Float[Array, "..."] = (
        jnp.ones(NUM_DIFFERENT_CARDS, dtype=int) * NUM_REPEAT_CARDS
    )


class LeducPoker(pgx.leduc_holdem.LeducHoldem, cfrx.envs.Env):
    @classmethod
    def action_to_string(cls, action: Int[Array, ""]) -> str:
        strings = ["c", "r", "f"]
        if action != -1:
            a = int(action)
            rep = strings[a]
        else:
            rep = "?"
        return rep

    @property
    def max_episode_length(self) -> int:
        return 12

    @property
    def max_nodes(self) -> int:
        return 2000

    @property
    def n_info_states(self) -> int:
        return len(INFO_SETS)

    @property
    def n_actions(self) -> int:
        return super().num_actions

    @property
    def n_players(self) -> int:
        return self.num_players

    def update_info_state(
        self, state: State, next_state: State, action: Int[Array, ""]
    ) -> InfoState:
        info_state = next_state.info_state
        assert info_state is not None
        private_card = next_state._cards[next_state.current_player]
        public_card = next_state._cards[-1]

        current_position = (info_state.action_sequence[state._round] != -1).sum()
        action_sequence = info_state.action_sequence.at[
            state._round, current_position
        ].set(action.astype(jnp.int8))
        updated_info_state = info_state._replace(action_sequence=action_sequence)

        info_state = jax.tree_map(
            lambda x, y: jnp.where(state.chance_node, x, y),
            info_state,
            updated_info_state,
        )

        return info_state._replace(
            private_card=private_card,
            public_card=public_card,
        )

    def info_state_to_str(self, info_state: InfoState) -> str:
        strings = ["c", "r", "f"]
        rep = f"{info_state.private_card}"

        for action in np.array(info_state.action_sequence)[0]:
            if action != -1:
                rep += strings[action]

        if info_state.public_card != -1:
            rep += f"{info_state.public_card}"
            for action in np.array(info_state.action_sequence)[1]:
                if action != -1:
                    rep += strings[action]
        return rep

    def info_state_idx(self, info_state: InfoState) -> Array:
        info_state_ravel = ravel(info_state)
        return reverse_array_lookup(info_state_ravel, INFO_SETS_VALUES)

    def _init(self, rng: Shaped[PRNGKeyArray, "2"]) -> State:
        env_state = super()._init(rng)
        info_state = InfoState(
            private_card=jnp.int8(-1),
            public_card=jnp.int8(-1),
            action_sequence=jnp.ones((2, 4), dtype=jnp.int8) * -1,
        )
        cards = jnp.int8([-1, -1, -1])
        return State(
            _first_player=env_state._first_player,
            current_player=env_state.current_player,
            observation=env_state.observation,
            rewards=env_state.rewards,
            terminated=env_state.terminated,
            truncated=env_state.truncated,
            _step_count=env_state._step_count,
            _last_action=env_state._last_action,
            _round=jnp.int8(0),
            _cards=cards,
            legal_action_mask=jnp.ones_like(env_state.legal_action_mask),
            _chips=env_state._chips,
            _raise_count=env_state._raise_count,
            info_state=info_state,
            chance_prior=jnp.ones(NUM_DIFFERENT_CARDS, dtype=int) * NUM_REPEAT_CARDS,
            chance_node=True,
        )

    def _resolve_decision_node(
        self, state: State, action: Int[Array, ""], random_key: PRNGKeyArray
    ) -> State:
        env_state = super()._step(state=state, action=action, key=random_key)

        is_public_card_unknown = env_state._cards[-1] == -1
        chance_node = (
            (env_state._round > 0) & is_public_card_unknown & ~env_state.terminated
        )

        legal_action_mask = jnp.where(
            chance_node,
            jnp.ones_like(env_state.legal_action_mask),
            env_state.legal_action_mask,
        )

        state = State(
            _first_player=env_state._first_player,
            current_player=env_state.current_player,
            observation=env_state.observation,
            rewards=env_state.rewards,
            terminated=env_state.terminated,
            truncated=env_state.truncated,
            _step_count=env_state._step_count,
            _last_action=env_state._last_action,
            _round=env_state._round.astype(jnp.int8),
            _cards=env_state._cards,
            legal_action_mask=legal_action_mask,
            _chips=env_state._chips,
            _raise_count=env_state._raise_count,
            info_state=env_state.info_state,
            chance_node=chance_node,
            chance_prior=env_state.chance_prior,
        )

        return state

    def _resolve_chance_node(
        self, state: State, action: Int[Array, ""], random_key: PRNGKeyArray
    ) -> State:
        draw_player = NUM_TOTAL_CARDS - state.chance_prior.sum()
        cards = state._cards.at[draw_player].set(action.astype(jnp.int8))
        chance_prior = state.chance_prior.at[action].add(-1)
        chance_node = (cards[:2] == -1).any()

        legal_action_mask = jnp.where(
            chance_node,
            jnp.ones_like(state.legal_action_mask),
            jnp.ones_like(state.legal_action_mask).at[-1].set(False),
        )
        return State(
            _first_player=state._first_player,
            current_player=state.current_player,
            observation=state.observation,
            rewards=state.rewards,
            terminated=state.terminated,
            truncated=state.truncated,
            _step_count=state._step_count,
            _last_action=state._last_action,
            _round=state._round.astype(jnp.int8),
            _cards=cards,
            legal_action_mask=legal_action_mask,
            _chips=state._chips,
            _raise_count=state._raise_count,
            info_state=state.info_state,
            chance_prior=chance_prior,
            chance_node=chance_node,
        )

    def _step(self, state: State, action: Array, random_key: PRNGKeyArray) -> State:
        new_state = jax.lax.cond(
            state.chance_node,
            lambda: self._resolve_chance_node(
                state=state, action=action, random_key=random_key
            ),
            lambda: self._resolve_decision_node(
                state=state, action=action, random_key=random_key
            ),
        )

        _round = jnp.where(new_state._cards[-1] != -1, jnp.maximum(state._round, 1), 0)
        new_state = new_state.replace(_round=_round)
        info_state = self.update_info_state(
            state=state, next_state=new_state, action=action
        )

        return new_state.replace(info_state=info_state)
