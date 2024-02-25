from __future__ import annotations

from typing import Optional

import jax
import jax.flatten_util
import jax.numpy as jnp
from jaxtyping import Array, Bool, Int, PyTree


def get_action_mask(state: PyTree) -> Bool[Array, "..."]:
    chance_action_mask = state.chance_prior > 0
    decision_action_mask = state.legal_action_mask
    max_action_mask_size = max(
        chance_action_mask.shape[-1], decision_action_mask.shape[-1]
    )
    chance_action_mask = jnp.pad(
        chance_action_mask,
        (0, max_action_mask_size - chance_action_mask.shape[-1]),
        constant_values=(False,),
    )
    decision_action_mask = jnp.pad(
        decision_action_mask,
        (0, max_action_mask_size - decision_action_mask.shape[-1]),
        constant_values=(False,),
    )
    action_mask = jnp.where(
        state.chance_node[..., None],
        chance_action_mask,
        decision_action_mask,
    )
    return action_mask


def reverse_array_lookup(x: Array, lookup_table: Array) -> Int[Array, ""]:
    return (lookup_table == x).all(axis=1).argmax()


def tree_unstack(tree: PyTree) -> list[PyTree]:
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    n_trees = leaves[0].shape[0]
    new_leaves: list = [[] for _ in range(n_trees)]
    for leaf in leaves:
        for i in range(n_trees):
            new_leaves[i].append(leaf[i])
    new_trees = [treedef.unflatten(leaves) for leaves in new_leaves]
    return new_trees


def ravel(tree: PyTree) -> Array:
    return jax.flatten_util.ravel_pytree(tree)[0]


def regret_matching(regrets: Array) -> Array:
    positive_regrets = jnp.maximum(regrets, 0)
    n_actions = positive_regrets.shape[-1]
    sum_pos_regret = positive_regrets.sum(axis=-1, keepdims=True)
    dist = jnp.where(
        sum_pos_regret == 0, 1 / n_actions, positive_regrets / sum_pos_regret
    )
    return dist


def log_array(x: Array, name: Optional[str] = None) -> None:
    if name is None:
        name = "array"
    jax.debug.print(name + ": {x}", x=x)
