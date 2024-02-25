import jax
import numpy as np
import pytest

from cfrx.algorithms.mccfr.outcome_sampling import (
    MCCFRState,
    compute_regrets_and_strategy_profile,
    unroll,
)
from cfrx.metrics import exploitability
from cfrx.policy import TabularPolicy
from cfrx.utils import regret_matching


@pytest.mark.parametrize(
    "env_name, num_iterations, target_exploitability",
    [
        ("Kuhn Poker", 20000, 15e-3),
        ("Leduc Poker", 50000, 5e-1),
    ],
)
def test_perf(env_name: str, num_iterations: int, target_exploitability: float):
    device = jax.devices("cpu")[0]

    random_key = jax.random.PRNGKey(0)
    if env_name == "Kuhn Poker":
        from cfrx.envs.kuhn_poker.constants import INFO_SETS
        from cfrx.envs.kuhn_poker.env import KuhnPoker

        env_cls = KuhnPoker
        EPISODE_LEN = 8
        NUM_MAX_NODES = 100

    elif env_name == "Leduc Poker":
        from cfrx.envs.leduc_poker.constants import INFO_SETS
        from cfrx.envs.leduc_poker.env import LeducPoker

        env_cls = LeducPoker
        EPISODE_LEN = 20
        NUM_MAX_NODES = 2000

    env = env_cls()

    n_states = len(INFO_SETS)
    n_actions = env.num_actions

    training_state = MCCFRState.init(n_states, n_actions)

    policy = TabularPolicy(
        n_actions=n_actions,
        exploration_factor=0.6,
        info_state_idx_fn=env.info_state_idx,
    )

    def do_iteration(training_state, random_key, env, policy, update_player):
        """
        Do one iteration of MCCFR: traverse the game tree once and
        compute counterfactual regrets and strategy profiles
        """

        random_key, subkey = jax.random.split(random_key)

        # Sample one path in the game tree
        random_key, subkey = jax.random.split(random_key)
        episode, states = unroll(
            init_state=env.init(subkey),
            training_state=training_state,
            random_key=subkey,
            update_player=update_player,
            env=env,
            policy=policy,
            n_max_steps=EPISODE_LEN,
        )

        # Compute counterfactual values and strategy profile
        (
            info_states,
            sampled_regrets,
            sampled_avg_probs,
        ) = compute_regrets_and_strategy_profile(
            episode=episode,
            training_state=training_state,
            policy=policy,
            update_player=update_player,
        )
        info_states_idx = jax.vmap(env.info_state_idx)(info_states)

        # Store regret and strategy profile values
        regrets = training_state.regrets.at[info_states_idx].add(sampled_regrets)
        avg_probs = training_state.avg_probs.at[info_states_idx].add(sampled_avg_probs)

        return regrets, avg_probs, episode

    do_iteration = jax.jit(
        do_iteration, static_argnames=("env", "policy"), device=device
    )

    # This function measures the exploitability of a strategy
    exploitability_fn = jax.jit(
        lambda policy_params: exploitability(
            policy_params=policy_params,
            env=env,
            n_players=2,
            n_max_nodes=NUM_MAX_NODES,
            policy=policy,
        ),
        device=device,
    )

    # One iteration consists in updating the policy for both players
    n_loops = 2 * num_iterations

    for k in range(n_loops):
        random_key, subkey = jax.random.split(random_key)

        # Update players alternatively
        update_player = k % 2
        new_regrets, new_avg_probs, episode = do_iteration(
            training_state,
            random_key,
            env=env,
            policy=policy,
            update_player=update_player,
        )

        # Accumulate regrets, compute new strategy and avg strategy
        new_probs = regret_matching(new_regrets)
        new_probs /= new_probs.sum(axis=-1, keepdims=True)

        training_state = training_state._replace(
            regrets=new_regrets,
            probs=new_probs,
            avg_probs=new_avg_probs,
            step=training_state.step + 1,
        )

    current_policy = training_state.avg_probs
    current_policy /= training_state.avg_probs.sum(axis=-1, keepdims=True)
    exp = exploitability_fn(policy_params=current_policy)

    assert np.allclose(exp, target_exploitability, rtol=1e-1, atol=1e-2)
