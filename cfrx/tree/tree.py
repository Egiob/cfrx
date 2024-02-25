from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PyTree


class Root(NamedTuple):
    """
    Base class to hold the root of a search tree.

    Args:
        prior_logits: `[n_actions]` the action prior logits.
        value: `[]` the value of the root node.
        state: `[...]` the state of the root node.
    """

    prior_logits: Array
    value: Array
    state: Array


UNVISITED = -1
NO_PARENT = -1
ROOT_INDEX = 0


class Tree(NamedTuple):
    """
    Adapted with minor modifications from https://github.com/google-deepmind/mctx.

    State of a search tree.

    The `Tree` dataclass is used to hold and inspect search data for a batch of
    inputs. In the fields below `N` represents
    the number of nodes in the tree, and `n_actions` is the number of discrete
    actions.

    node_visits: `[N]` the visit counts for each node.
    raw_values: `[N, n_players]` the raw value for each node.
    node_values: `[N, n_players]` the cumulative search value for each node.
    parents: `[N]` the node index for the parents for each node.
    action_from_parent: `[N]` action to take from the parent to reach each
      node.
    children_index: `[N, n_actions]` the node index of the children for each
      action.
    children_prior_logits: `[N, n_actions` the action prior logits of each
      node.
    children_visits: `[N, n_actions]` the visit counts for children for
      each action.
    children_rewards: `[N, n_actions, n_players]` the immediate reward for each
      action.
    children_values: `[N, n_actions, n_players]` the value of the next node after
      the action.
    states: `[N, ...]` the state embeddings of each node.
    depth: `[N]` the depth of each node in the tree.
    extra_data: `[...]` extra data passed to the tree.

    """

    node_visits: Int[Array, "..."]
    raw_values: Float[Array, "... n_players"]
    node_values: Float[Array, "... n_players"]
    parents: Int[Array, "..."]
    action_from_parent: Int[Array, "..."]
    children_index: Int[Array, "... n_actions"]
    children_prior_logits: Float[Array, "... n_actions"]
    children_visits: Int[Array, "... n_actions"]
    children_rewards: Float[Array, "... n_actions n_players"]
    children_values: Float[Array, "... n_actions n_players"]
    states: PyTree
    depth: Int[Array, "..."]
    extra_data: dict[str, Array]

    @classmethod
    @property
    def ROOT_INDEX(cls) -> Int[Array, ""]:
        return jnp.asarray(ROOT_INDEX)

    @classmethod
    @property
    def NO_PARENT(cls) -> Int[Array, ""]:
        return jnp.asarray(NO_PARENT)

    @classmethod
    @property
    def UNVISITED(cls) -> Int[Array, ""]:
        return jnp.asarray(UNVISITED)

    @property
    def n_actions(self) -> int:
        return self.children_index.shape[-1]
