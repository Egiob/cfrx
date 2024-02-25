from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from cfrx.tree import Tree


def backward_one_info_set(
    tree: Tree,
    info_states: Array,
    current_info_state: Int[Array, ""],
    br_player: int,
    depth: Int[Array, ""],
) -> Tree:
    # Select all nodes in this infoset
    ntn_at_depth = (tree.depth == depth) & (~tree.states.terminated)
    selected_info_set_mask = (info_states == current_info_state) & ntn_at_depth

    is_br_player = (tree.states.current_player == br_player) & ~tree.states.chance_node

    is_br_player = (is_br_player * selected_info_set_mask).sum() > 0

    legal_action_mask = (
        tree.states.legal_action_mask * selected_info_set_mask[..., None]
    ).sum(axis=0) > 0

    p_opponent = tree.extra_data["p_opponent"]
    p_chance = tree.extra_data["p_chance"]

    # Get expected values for each of the node in the infoset
    cf_reach_prob = p_opponent[..., None] * p_chance[..., None]

    best_response_values = jnp.where(
        selected_info_set_mask[..., None],
        tree.children_values[..., br_player] * cf_reach_prob,
        jnp.nan,
    )

    best_response_values = jnp.nansum(best_response_values, axis=0)
    best_action = jnp.where(legal_action_mask, best_response_values, -jnp.inf).argmax()

    best_response_current_value = tree.children_values[:, best_action, br_player]

    expected_current_value = (
        tree.children_values[..., br_player] * tree.children_prior_logits
    ).sum(axis=1)

    current_value = jnp.where(
        is_br_player,
        best_response_current_value,
        expected_current_value,
    )

    new_node_values = jnp.where(
        selected_info_set_mask,
        current_value,
        tree.node_values[..., br_player],
    )

    new_children_values = jnp.where(
        tree.children_index != -1, new_node_values[tree.children_index], 0
    )

    tree = tree._replace(
        node_values=tree.node_values.at[..., br_player].set(new_node_values),
        children_values=tree.children_values.at[..., br_player].set(new_children_values),
    )

    return tree


def backward_one_depth_level(
    tree: Tree,
    depth: Int[Array, ""],
    br_player: int,
    info_state_fn: Callable,
) -> Tree:
    info_states = jax.vmap(info_state_fn)(tree.states.info_state)

    def cond_fn(val: tuple[Tree, Array]) -> Array:
        tree, backward_visited = val
        ntn_at_depth = (tree.depth == depth) & (~tree.states.terminated)

        count_ntn_to_visit = jnp.where(ntn_at_depth & ~backward_visited, 1, 0).sum()

        return count_ntn_to_visit > 0

    def loop_fn(val: tuple[Tree, Array]) -> tuple[Tree, Array]:
        tree, backward_visited = val
        ntn_at_depth = (tree.depth == depth) & (~tree.states.terminated)

        ntn_to_visit = jnp.where(ntn_at_depth & ~backward_visited, info_states, -1)

        # Select an infoset to reduce
        selected_infoset_idx = ntn_to_visit.argmax()
        selected_info_state = info_states[selected_infoset_idx]

        tree = backward_one_info_set(
            tree=tree,
            depth=depth,
            info_states=info_states,
            current_info_state=selected_info_state,
            br_player=br_player,
        )

        selected_info_set_mask = info_states == selected_info_state
        backward_visited = jnp.where(
            selected_info_set_mask, jnp.bool_(True), backward_visited
        )

        return tree, backward_visited

    n_max_nodes = tree.node_values.shape[0]
    backward_visited = jnp.zeros(n_max_nodes).astype(bool)

    tree, backward_visited = jax.lax.while_loop(
        cond_fn, loop_fn, (tree, backward_visited)
    )

    return tree


def compute_best_response_value(
    tree: Tree,
    br_player: int,
    info_state_fn: Callable,
) -> Float[Array, " num_players"]:
    depth = tree.depth.max()

    def cond_fn(val: tuple[Tree, Array]) -> Array:
        _, depth = val
        return depth >= 0

    def loop_fn(val: tuple[Tree, Array]) -> tuple[Tree, Array]:
        tree, depth = val
        tree = backward_one_depth_level(
            tree=tree, depth=depth, br_player=br_player, info_state_fn=info_state_fn
        )
        depth -= 1
        return tree, depth

    tree, _ = jax.lax.while_loop(cond_fn, loop_fn, (tree, depth))
    return tree.node_values[0]
