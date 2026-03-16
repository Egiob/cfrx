import functools as ft
from typing import Tuple

import jax
import jax.numpy as jnp
import pgx
from jaxtyping import Array, Bool, Float, Int, PyTree

from cfrx.policy import Policy
from cfrx.tree import Root, Tree
from cfrx.utils import get_action_mask


def instantiate_tree_from_root(
    root: Root,
    n_max_nodes: int,
    n_players: int,
    running_probabilities: bool = False,
) -> Tree:
    """Initializes tree state at search root."""
    (n_actions,) = root.prior_logits.shape

    data_dtype = root.value.dtype

    def _zeros(x: Array) -> Array:
        return jnp.zeros((n_max_nodes,) + x.shape, dtype=x.dtype)

    # Create a new empty tree state and fill its root.
    tree = Tree(
        node_visits=jnp.zeros(n_max_nodes, dtype=jnp.int32),
        raw_values=jnp.zeros((n_max_nodes, n_players), dtype=data_dtype),
        node_values=jnp.zeros((n_max_nodes, n_players), dtype=data_dtype),
        parents=jnp.full(
            n_max_nodes,
            Tree.NO_PARENT,
            dtype=jnp.int32,
        ),
        action_from_parent=jnp.full(
            n_max_nodes,
            Tree.NO_PARENT,
            dtype=jnp.int32,
        ),
        children_index=jnp.full(
            (n_max_nodes, n_actions),
            Tree.UNVISITED,
            dtype=jnp.int32,
        ),
        children_prior_logits=jnp.zeros(
            (n_max_nodes, n_actions), dtype=root.prior_logits.dtype
        ),
        children_values=jnp.zeros((n_max_nodes, n_actions, n_players), dtype=data_dtype),
        children_visits=jnp.zeros((n_max_nodes, n_actions), dtype=jnp.int32),
        children_rewards=jnp.zeros(
            (n_max_nodes, n_actions, n_players), dtype=data_dtype
        ),
        to_visit=jnp.zeros(n_max_nodes, dtype=bool),
        states=jax.tree_util.tree_map(_zeros, root.state),
        depth=jnp.ones(n_max_nodes, dtype=jnp.int32) * -1,
        extra_data={},
    )
    new_tree: Tree = tree._replace(
        node_visits=tree.node_visits.at[Tree.ROOT_INDEX].set(1),
        states=jax.tree_map(
            lambda x, y: x.at[Tree.ROOT_INDEX].set(y), tree.states, root.state
        ),
        children_prior_logits=tree.children_prior_logits.at[Tree.ROOT_INDEX].set(
            root.prior_logits
        ),
        depth=tree.depth.at[Tree.ROOT_INDEX].set(0),
    )

    if running_probabilities:
        new_tree = initialize_running_probabilities(new_tree)

    return new_tree


def initialize_running_probabilities(tree: Tree) -> Tree:
    n_max_nodes = tree.node_visits.shape[-1]
    init_prob = (jnp.ones(n_max_nodes) * -1).at[Tree.ROOT_INDEX].set(1.0)
    running_probabilities = {
        "p_self": init_prob,
        "p_opponent": init_prob,
        "p_chance": init_prob,
    }
    tree = tree._replace(extra_data={**tree.extra_data, **running_probabilities})
    return tree


def add_children(
    tree: Tree,
    state: PyTree,
    env: pgx.Env,
    node_counter: jax.Array,
    parent_idx: jax.Array,
) -> tuple[Tree, jax.Array]:

    chance_fn = ft.partial(add_chance_children, env=env)
    player_fn = ft.partial(add_player_children, env=env)
    return jax.lax.cond(
        state.chance_node, chance_fn, player_fn, tree, state, node_counter, parent_idx
    )


def add_player_children(
    tree: Tree,
    state: PyTree,
    node_counter: jax.Array,
    parent_idx: jax.Array,
    env: pgx.Env,
) -> tuple[Tree, jax.Array]:

    action_mask = env.get_action_mask(state) & ~state.terminated
    n_actions = len(action_mask)
    n_max_nodes = len(tree.to_visit)
    action_idx = jnp.where(action_mask, jnp.arange(n_actions), n_max_nodes)
    action_idx = jnp.sort(action_idx)

    update_idx = jnp.arange(n_actions) + node_counter + 1
    update_idx = jnp.where(action_idx == n_max_nodes, n_max_nodes, update_idx)

    action_from_parent = tree.action_from_parent.at[update_idx].set(action_idx)
    parents = tree.parents.at[update_idx].set(parent_idx)
    to_visit = tree.to_visit.at[update_idx].set(True)

    tree = tree._replace(
        action_from_parent=action_from_parent, parents=parents, to_visit=to_visit
    )

    return tree, action_mask.sum()


def add_chance_children(
    tree: Tree,
    state: PyTree,
    node_counter: jax.Array,
    parent_idx: jax.Array,
    env: pgx.Env,
) -> tuple[Tree, jax.Array]:

    action_mask = env.get_chance_mask(state) & ~state.terminated
    n_actions = len(action_mask)
    n_max_nodes = len(tree.to_visit)
    action_idx = jnp.where(action_mask, jnp.arange(n_actions), n_max_nodes)
    action_idx = jnp.sort(action_idx)

    update_idx = jnp.arange(n_actions) + node_counter + 1
    update_idx = jnp.where(action_idx == n_max_nodes, n_max_nodes, update_idx)

    action_from_parent = tree.action_from_parent.at[update_idx].set(action_idx)
    parents = tree.parents.at[update_idx].set(parent_idx)
    to_visit = tree.to_visit.at[update_idx].set(True)

    tree = tree._replace(
        action_from_parent=action_from_parent, parents=parents, to_visit=to_visit
    )

    return tree, action_mask.sum()


def select_new_node_and_play(
    tree: Tree, env: pgx.Env
) -> tuple[PyTree, jax.Array, jax.Array, jax.Array]:

    child_index = jnp.argmax(tree.to_visit)
    parent_index = tree.parents[child_index]
    action = tree.action_from_parent[child_index]

    parent_state = jax.tree_map(lambda x: x[parent_index], tree.states)
    new_state = env.step(parent_state, action)

    return new_state, parent_index, child_index, action


def update_running_probabilities(
    tree: Tree,
    parent_index: jax.Array,
    next_node_index: jax.Array,
    strategy: Float[Array, ""],
    traverser: int,
) -> Tree:
    parent_state = jax.tree_map(lambda x: x[parent_index], tree.states)

    p_self = tree.extra_data["p_self"]

    update_p_self_condition = (
        (parent_state.current_player == traverser)
        & (~parent_state.terminated)
        & (~parent_state.chance_node)
    )
    p_self_new_value = jnp.where(
        update_p_self_condition,
        p_self[parent_index] * strategy,
        p_self[parent_index],
    )

    p_opponent = tree.extra_data["p_opponent"]

    update_p_opponent_condition = (
        (parent_state.current_player != traverser)
        & (~parent_state.terminated)
        & (~parent_state.chance_node)
    )

    p_opponent_new_value = jnp.where(
        update_p_opponent_condition,
        p_opponent[parent_index] * strategy,
        p_opponent[parent_index],
    )

    p_chance = tree.extra_data["p_chance"]

    p_chance_new_value = jnp.where(
        parent_state.chance_node,
        p_chance[parent_index] * strategy,
        p_chance[parent_index],
    )

    tree = tree._replace(
        extra_data={
            **tree.extra_data,
            **{
                "p_self": p_self.at[next_node_index].set(p_self_new_value),
                "p_opponent": p_opponent.at[next_node_index].set(p_opponent_new_value),
                "p_chance": p_chance.at[next_node_index].set(p_chance_new_value),
            },
        }
    )
    return tree


def traverse_tree_vanilla(
    tree: Tree,
    env: pgx.Env,
) -> Tree:
    def cond_fn(val: Tuple) -> Bool[Array, ""]:
        tree, n = val
        n_max_nodes = len(tree.node_visits)

        return tree.to_visit.any() & (n < n_max_nodes)

    def loop_fn(val: Tuple) -> Tuple:
        tree, n = val

        new_state, parent_index, child_index, action = select_new_node_and_play(
            tree, env
        )

        tree, n_added = add_children(
            tree=tree, state=new_state, env=env, node_counter=n, parent_idx=child_index
        )

        tree = tree._replace(
            node_visits=tree.node_visits.at[child_index].set(1),
            node_values=tree.node_values.at[child_index].set(new_state.rewards),
            states=jax.tree_map(
                lambda x, y: x.at[child_index].set(y), tree.states, new_state
            ),
            raw_values=tree.raw_values.at[child_index].set(new_state.rewards),
            children_index=tree.children_index.at[parent_index, action].set(child_index),
            children_rewards=tree.children_rewards.at[parent_index, action].set(
                new_state.rewards
            ),
            children_values=tree.children_values.at[parent_index, action].set(
                new_state.rewards
            ),
            parents=tree.parents.at[child_index].set(parent_index),
            action_from_parent=tree.action_from_parent.at[child_index].set(action),
            depth=tree.depth.at[child_index].set(tree.depth[parent_index] + 1),
            to_visit=tree.to_visit.at[child_index].set(False),
        )

        return tree, n + n_added

    tree, n_added = add_children(
        tree=tree,
        state=jax.tree_map(lambda x: x[0], tree.states),
        env=env,
        node_counter=0,
        parent_idx=0,
    )

    tree, _ = jax.lax.while_loop(cond_fn, loop_fn, (tree, n_added))
    return tree


def traverse_tree_cfr(
    tree: Tree,
    policy: Policy,
    policy_params: Array,
    env: pgx.Env,
    traverser: int = 0,
) -> Tree:
    def cond_fn(val: Tuple) -> Bool[Array, ""]:
        tree, n = val
        n_max_nodes = len(tree.node_visits)

        return tree.to_visit.any() & (n < n_max_nodes)

    def loop_fn(val: Tuple) -> Tuple:
        tree, n = val

        new_state, parent_index, child_index, action = select_new_node_and_play(
            tree, env
        )

        tree, n_added = add_children(
            tree=tree, state=new_state, env=env, node_counter=n, parent_idx=child_index
        )

        parent_state = jax.tree_map(lambda x: x[parent_index], tree.states)

        strategy = policy.prob_distribution(
            params=policy_params,
            info_state=env.get_info_state(parent_state),
            action_mask=env.get_action_mask(state=parent_state),
            use_behavior_policy=jnp.bool_(False),
        )

        chance_probs = env.get_chance_probs(parent_state)
        chance_strategy = chance_probs[action]

        action_prob = jnp.where(
            parent_state.chance_node, chance_strategy, strategy[action]
        )

        tree = update_running_probabilities(
            tree=tree,
            parent_index=parent_index,
            next_node_index=child_index,
            strategy=action_prob,
            traverser=traverser,
        )

        tree = tree._replace(
            node_visits=tree.node_visits.at[child_index].set(1),
            node_values=tree.node_values.at[child_index].set(new_state.rewards),
            states=jax.tree_map(
                lambda x, y: x.at[child_index].set(y), tree.states, new_state
            ),
            raw_values=tree.raw_values.at[child_index].set(new_state.rewards),
            children_index=tree.children_index.at[parent_index, action].set(child_index),
            children_rewards=tree.children_rewards.at[parent_index, action].set(
                new_state.rewards
            ),
            children_values=tree.children_values.at[parent_index, action].set(
                new_state.rewards
            ),
            children_prior_logits=tree.children_prior_logits.at[parent_index].set(
                jnp.where(parent_state.chance_node, chance_probs, strategy)
            ),
            parents=tree.parents.at[child_index].set(parent_index),
            action_from_parent=tree.action_from_parent.at[child_index].set(action),
            depth=tree.depth.at[child_index].set(tree.depth[parent_index] + 1),
            to_visit=tree.to_visit.at[child_index].set(False),
        )

        return tree, n + n_added

    tree, n_added = add_children(
        tree=tree,
        state=jax.tree_map(lambda x: x[0], tree.states),
        env=env,
        node_counter=0,
        parent_idx=0,
    )

    tree, _ = jax.lax.while_loop(cond_fn, loop_fn, (tree, n_added))
    return tree
