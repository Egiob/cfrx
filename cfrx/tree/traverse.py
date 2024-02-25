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
        states=jax.tree_util.tree_map(_zeros, root.state),
        depth=jnp.ones(n_max_nodes, dtype=jnp.int32) * -1,
        extra_data={},
    )
    new_tree: Tree = tree._replace(
        node_visits=tree.node_visits.at[Tree.ROOT_INDEX].set(1),
        # node_values=tree.node_values.at[Tree.ROOT_INDEX].set(root.value),
        states=jax.tree_map(
            lambda x, y: x.at[Tree.ROOT_INDEX].set(y), tree.states, root.state
        ),
        # raw_values=tree.node_values.at[Tree.ROOT_INDEX].set(root.value),
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


def select_new_node_and_play(
    tree: Tree, env: pgx.Env
) -> tuple[PyTree, Int[Array, ""], Int[Array, ""]]:
    action_mask = jax.vmap(get_action_mask)(tree.states)

    is_to_visit = (
        (action_mask * (1 - tree.states.terminated)[..., None] * tree.children_index) < 0
    ).any(axis=1)

    parent_index = jnp.argmax(is_to_visit)

    unexplored_actions = jnp.where(
        tree.children_index[parent_index] == Tree.UNVISITED,
        action_mask[parent_index] & ~tree.states.terminated[parent_index],
        False,
    )

    action = jnp.argmax(unexplored_actions)

    new_state = env.step(jax.tree_map(lambda x: x[parent_index], tree.states), action)

    return new_state, parent_index, action


def update_running_probabilities(
    tree: Tree,
    parent_index: Int[Array, ""],
    next_node_index: Int[Array, ""],
    strategy: Float[Array, ""],
    env: pgx.Env,
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
        action_mask = jax.vmap(get_action_mask)(tree.states)
        is_to_visit = (
            (action_mask * (1 - tree.states.terminated)[..., None] * tree.children_index)
            < 0
        ).any(axis=1)

        return jnp.logical_and(is_to_visit.any(), (n < n_max_nodes))

    def loop_fn(val: Tuple) -> Tuple:
        tree, n = val

        n += 1
        next_node_index = n

        new_state, parent_index, action = select_new_node_and_play(tree, env)

        parent_state = jax.tree_map(lambda x: x[parent_index], tree.states)

        action_mask = get_action_mask(parent_state)

        strategy = action_mask / action_mask.sum()

        tree = tree._replace(
            node_visits=tree.node_visits.at[next_node_index].set(1),
            node_values=tree.node_values.at[next_node_index].set(new_state.rewards),
            states=jax.tree_map(
                lambda x, y: x.at[next_node_index].set(y),
                tree.states,
                new_state,
            ),
            raw_values=tree.raw_values.at[next_node_index].set(new_state.rewards),
            children_prior_logits=tree.children_prior_logits.at[parent_index].set(
                strategy
            ),
            children_index=tree.children_index.at[parent_index, action].set(
                next_node_index
            ),
            children_rewards=tree.children_rewards.at[parent_index, action].set(
                new_state.rewards
            ),
            children_values=tree.children_values.at[parent_index, action].set(
                new_state.rewards
            ),
            parents=tree.parents.at[next_node_index].set(parent_index),
            action_from_parent=tree.action_from_parent.at[next_node_index].set(action),
            depth=tree.depth.at[next_node_index].set(tree.depth[parent_index] + 1),
        )

        return tree, n

    tree, _ = jax.lax.while_loop(cond_fn, loop_fn, (tree, 0))
    return tree


def traverse_tree_cfr(
    tree: Tree,
    policy: Policy,
    policy_params: Array,
    env: pgx.Env,
    traverser: int = 0,
) -> Tree:
    def cond_fn(val: tuple[Tree, Int[Array, ""]]) -> Bool[Array, ""]:
        tree, n = val
        n_max_nodes = len(tree.node_visits)
        action_mask = jax.vmap(get_action_mask)(tree.states)
        is_to_visit = (
            (action_mask * (1 - tree.states.terminated)[..., None] * tree.children_index)
            < 0
        ).any(axis=1)

        return jnp.logical_and(is_to_visit.any(), (n < n_max_nodes))

    def loop_fn(val: tuple[Tree, Int[Array, ""]]) -> tuple[Tree, Int[Array, ""]]:
        tree, n = val

        n += 1
        next_node_index = n

        new_state, parent_index, action = select_new_node_and_play(tree, env)

        parent_state = jax.tree_map(lambda x: x[parent_index], tree.states)

        n_actions = len(get_action_mask(parent_state))

        strategy = policy.prob_distribution(
            params=policy_params,
            info_state=parent_state.info_state,
            action_mask=parent_state.legal_action_mask,
            use_behavior_policy=jnp.bool_(False),
        )
        strategy = jnp.pad(strategy, (0, n_actions - len(strategy)))

        chance_strategy = parent_state.chance_prior / parent_state.chance_prior.sum()
        chance_strategy = jnp.pad(chance_strategy, (0, n_actions - len(chance_strategy)))

        current_strategy = jnp.where(
            parent_state.chance_node,
            chance_strategy,
            strategy,
        )

        tree = update_running_probabilities(
            tree=tree,
            parent_index=parent_index,
            next_node_index=next_node_index,
            strategy=current_strategy[action],
            env=env,
            traverser=traverser,
        )

        tree = tree._replace(
            node_visits=tree.node_visits.at[next_node_index].set(1),
            node_values=tree.node_values.at[next_node_index].set(new_state.rewards),
            states=jax.tree_map(
                lambda x, y: x.at[next_node_index].set(y),
                tree.states,
                new_state,
            ),
            raw_values=tree.raw_values.at[next_node_index].set(new_state.rewards),
            children_prior_logits=tree.children_prior_logits.at[parent_index].set(
                current_strategy
            ),
            children_index=tree.children_index.at[parent_index, action].set(
                next_node_index
            ),
            children_rewards=tree.children_rewards.at[parent_index, action].set(
                new_state.rewards
            ),
            children_values=tree.children_values.at[parent_index, action].set(
                new_state.rewards
            ),
            parents=tree.parents.at[next_node_index].set(parent_index),
            action_from_parent=tree.action_from_parent.at[next_node_index].set(action),
            depth=tree.depth.at[next_node_index].set(tree.depth[parent_index] + 1),
        )

        return tree, n

    tree, _ = jax.lax.while_loop(cond_fn, loop_fn, (tree, 0))
    return tree
