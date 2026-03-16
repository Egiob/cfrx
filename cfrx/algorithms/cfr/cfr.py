from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from cfrx.envs import Env
from cfrx.policy import TabularPolicy
from cfrx.tree import Tree
from cfrx.tree.traverse import instantiate_tree_from_root, traverse_tree_cfr
from cfrx.tree.tree import Root
from cfrx.utils import regret_matching


class CFRState(NamedTuple):
    regrets: Float[Array, "*batch a"]
    probs: Float[Array, "... a"]
    avg_probs: Float[Array, "... a"]
    step: Int[Array, "..."]

    @classmethod
    def init(cls, n_states: int, n_actions: int) -> CFRState:
        return CFRState(
            regrets=jnp.zeros((n_states, n_actions)),
            probs=jnp.ones((n_states, n_actions))
            / jnp.ones((n_states, n_actions)).sum(axis=-1, keepdims=True),
            avg_probs=jnp.zeros((n_states, n_actions)) + 1e-6,
            step=jnp.array(1, dtype=int),
        )


def backward_one_infoset(
    tree: Tree,
    info_states: Array,
    current_infoset: Int[Array, ""],
    cfr_player: int,
    depth: Int[Array, ""],
) -> Tree:
    # Select all nodes in this infoset
    infoset_mask = (
        (tree.depth == depth)
        & (~tree.states.terminated)
        & (info_states == current_infoset)
    )

    is_cfr_player = (
        (tree.states.current_player == cfr_player)
        & (~tree.states.chance_node)
        & infoset_mask
    ).any()

    p_opponent = tree.extra_data["p_opponent"]
    p_chance = tree.extra_data["p_chance"]
    p_self = tree.extra_data["p_self"]

    # Get expected values for each of the node in the infoset
    cf_reach_prob = p_opponent * p_chance

    legal_action_mask = (tree.states.legal_action_mask & infoset_mask[..., None]).any(
        axis=0
    )

    cf_state_action_values = (
        tree.children_values[..., cfr_player]
        * cf_reach_prob[..., None]
        * legal_action_mask
    )  # (T, a)
    cf_state_action_values = jnp.sum(
        cf_state_action_values, axis=0, where=infoset_mask[..., None]  # (a,)
    )

    expected_values = (
        tree.children_values[..., cfr_player] * tree.children_prior_logits
    ).sum(axis=1)

    cf_state_values = jnp.sum(
        expected_values * cf_reach_prob, axis=0, where=infoset_mask
    )  # (,)

    new_node_values = jnp.where(
        infoset_mask, expected_values, tree.node_values[..., cfr_player]
    )

    new_children_values = jnp.where(
        tree.children_index != -1, new_node_values[tree.children_index], 0
    )

    regrets = (
        cf_state_action_values - cf_state_values[..., None]
    ) * legal_action_mask  # (a,)

    strategy_profile = jnp.sum(
        p_self[..., None] * tree.children_prior_logits,
        axis=0,
        where=infoset_mask[..., None],
    )  # (a,)

    tree = tree._replace(
        node_values=tree.node_values.at[..., cfr_player].set(new_node_values),
        children_values=tree.children_values.at[..., cfr_player].set(
            new_children_values
        ),
        extra_data={
            **tree.extra_data,
            "regrets": jnp.where(
                infoset_mask[..., None] & is_cfr_player,
                tree.extra_data["regrets"] + regrets,
                tree.extra_data["regrets"],
            ),
            "strategy_profile": jnp.where(
                infoset_mask[..., None] & is_cfr_player,
                tree.extra_data["strategy_profile"] + strategy_profile,
                tree.extra_data["strategy_profile"],
            ),
        },
    )

    return tree


def backward_one_depth_level(
    tree: Tree,
    depth: Int[Array, ""],
    cfr_player: int,
    info_state_fn: Callable,
) -> Tree:
    info_states = jax.vmap(info_state_fn)(tree.states.info_state)

    def cond_fn(val: tuple[Tree, Array]) -> Array:
        tree, visited = val
        nodes_to_visit_idx = (
            (tree.depth == depth) & (~tree.states.terminated) & (~visited)
        )
        return nodes_to_visit_idx.any()

    def loop_fn(val: tuple[Tree, Array]) -> tuple[Tree, Array]:
        tree, visited = val
        nodes_to_visit_idx = (
            (tree.depth == depth) & (~tree.states.terminated) & (~visited)
        )

        # Select an infoset to resolve
        selected_infoset_idx = nodes_to_visit_idx.argmax()
        selected_infoset = info_states[selected_infoset_idx]

        tree = backward_one_infoset(
            tree=tree,
            depth=depth,
            info_states=info_states,
            current_infoset=selected_infoset,
            cfr_player=cfr_player,
        )

        visited = jnp.where(info_states == selected_infoset, True, visited)
        return tree, visited

    visited = jnp.zeros(tree.node_values.shape[0], dtype=bool)
    tree, visited = jax.lax.while_loop(cond_fn, loop_fn, (tree, visited))

    return tree


def backward_cfr(
    tree: Tree,
    cfr_player: int,
    info_state_fn: Callable,
) -> Tree:
    depth = tree.depth.max()

    def cond_fn(val: tuple[Tree, Array]) -> Array:
        _, depth = val
        return depth >= 0

    def loop_fn(val: tuple[Tree, Array]) -> tuple[Tree, Array]:
        tree, depth = val
        tree = backward_one_depth_level(
            tree=tree, depth=depth, cfr_player=cfr_player, info_state_fn=info_state_fn
        )
        depth -= 1
        return tree, depth

    tree, _ = jax.lax.while_loop(cond_fn, loop_fn, (tree, depth))
    return tree


def do_iteration(
    training_state: CFRState,
    update_player: Int[Array, ""],
    env: Env,
    policy: TabularPolicy,
) -> CFRState:
    s0 = env.init(jax.random.PRNGKey(0))
    root = Root(
        prior_logits=s0.legal_action_mask * 1.0,
        value=jnp.zeros(1, dtype=float),
        state=s0,
    )

    tree = instantiate_tree_from_root(
        root,
        n_max_nodes=env.max_nodes,
        n_players=env.n_players,
        running_probabilities=True,
    )

    tree = traverse_tree_cfr(
        tree,
        policy=policy,
        policy_params=training_state.probs,
        env=env,
        traverser=update_player,
    )

    infoset_idx = jax.vmap(env.info_state_idx)(tree.states.info_state)
    squash_idx = jnp.unique(infoset_idx, size=env.n_info_states, return_index=True)[1]

    tree_regrets = training_state.regrets.take(infoset_idx, axis=0)
    tree_strategies = training_state.avg_probs.take(infoset_idx, axis=0)

    tree = tree._replace(
        extra_data={
            **tree.extra_data,
            "regrets": tree_regrets,
            "strategy_profile": tree_strategies,
        }
    )

    tree = backward_cfr(
        tree=tree, cfr_player=update_player, info_state_fn=env.info_state_idx
    )

    regrets = tree.extra_data["regrets"][squash_idx]
    strategy_profile = tree.extra_data["strategy_profile"][squash_idx]
    probs = regret_matching(regrets)
    new_state = training_state._replace(
        regrets=regrets,
        probs=probs,
        avg_probs=strategy_profile,
        step=training_state.step + 1,
    )
    return new_state
