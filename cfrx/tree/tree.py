"""A data structure used to hold / inspect search data for a batch of inputs."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from flax.struct import PyTreeNode
from jaxtyping import Array, Float, Int


class Root(PyTreeNode):
    """
    Base class to hold the root of a search tree.

    Args:
        prior_logits: `[num_actions]` the action prior logits.
        value: `[]` the value of the root node.
        state: `[...]` the state of the root node.
    """

    prior_logits: Array
    value: Array
    state: Array


UNVISITED = -1
NO_PARENT = -1
ROOT_INDEX = 0


class Tree(PyTreeNode):
    """
    Adapted with minor modifications from https://github.com/google-deepmind/mctx.

    State of a search tree.

    The `Tree` dataclass is used to hold and inspect search data for a batch of
    inputs. In the fields below `N` represents
    the number of nodes in the tree, and `num_actions` is the number of discrete
    actions.

    node_visits: `[N]` the visit counts for each node.
    raw_values: `[N, num_players]` the raw value for each node.
    node_values: `[N, num_players]` the cumulative search value for each node.
    parents: `[N]` the node index for the parents for each node.
    action_from_parent: `[N]` action to take from the parent to reach each
      node.
    children_index: `[N, num_actions]` the node index of the children for each
      action.
    children_prior_logits: `[N, num_actions` the action prior logits of each
      node.
    children_visits: `[N, num_actions]` the visit counts for children for
      each action.
    children_rewards: `[N, num_actions, num_players]` the immediate reward for each
      action.
    children_values: `[N, num_actions, num_players]` the value of the next node after
      the action.
    states: `[N, ...]` the state embeddings of each node.
    depth: `[N]` the depth of each node in the tree.
    extra_data: `[...]` extra data passed to the search.

    """

    node_visits: Int[Array, "..."]
    raw_values: Float[Array, "... num_players"]
    node_values: Float[Array, "... num_players"]
    parents: Int[Array, "..."]
    action_from_parent: Int[Array, "..."]
    children_index: Int[Array, "... num_actions"]
    children_prior_logits: Float[Array, "... num_actions"]
    children_visits: Int[Array, "... num_actions"]
    children_rewards: Float[Array, "... num_actions num_players"]
    children_values: Float[Array, "... num_actions num_players"]
    states: Any
    depth: Int[Array, "..."]
    extra_data: dict

    @classmethod  # type: ignore
    @property
    def ROOT_INDEX(cls) -> Int[Array, ""]:
        return jnp.asarray(ROOT_INDEX)

    @classmethod  # type: ignore
    @property
    def NO_PARENT(cls) -> Int[Array, ""]:
        return jnp.asarray(NO_PARENT)

    @classmethod  # type: ignore
    @property
    def UNVISITED(cls) -> Int[Array, ""]:
        return jnp.asarray(UNVISITED)

    @property
    def num_actions(self) -> int:
        return self.children_index.shape[-1]  # type: ignore
