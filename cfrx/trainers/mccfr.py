import jax
from jaxtyping import Array, Int, PRNGKeyArray
from tqdm import tqdm

from cfrx.algorithms.mccfr.outcome_sampling import MCCFRState, do_iteration
from cfrx.envs import Env
from cfrx.metrics import exploitability
from cfrx.policy import TabularPolicy


class MCCFRTrainer:
    def __init__(self, env: Env, policy: TabularPolicy):
        self._env = env
        self._policy = policy

        self._exploitability_fn = jax.jit(
            lambda policy_params: exploitability(
                policy_params=policy_params,
                env=env,
                n_players=env.n_players,
                n_max_nodes=env.max_nodes,
                policy=policy,
            )
        )

        self._do_iteration_fn = jax.jit(
            lambda training_state, random_key, update_player: do_iteration(
                training_state=training_state,
                random_key=random_key,
                env=env,
                policy=policy,
                update_player=update_player,
            )
        )

    def do_n_iterations(
        self,
        training_state: MCCFRState,
        update_player: Int[Array, ""],
        random_key: PRNGKeyArray,
        n: int,
    ) -> tuple[MCCFRState, Int[Array, ""]]:
        def _scan_fn(carry, unused):
            training_state, random_key, update_player = carry

            random_key, subkey = jax.random.split(random_key)
            update_player = (update_player + 1) % 2
            training_state = self._do_iteration_fn(
                training_state,
                subkey,
                update_player=update_player,
            )

            return (training_state, random_key, update_player), None

        (new_training_state, _, last_update_player), _ = jax.lax.scan(
            _scan_fn,
            (training_state, random_key, update_player),
            None,
            length=n,
        )

        return new_training_state, last_update_player

    def train(
        self, n_iterations: int, metrics_period: int, random_key: PRNGKeyArray
    ) -> MCCFRState:
        training_state = MCCFRState.init(self._env.n_info_states, self._env.n_actions)

        assert n_iterations % metrics_period == 0

        n_loops = n_iterations // metrics_period
        update_player = 0
        _do_n_iterations = jax.jit(
            lambda training_state, update_player, random_key: self.do_n_iterations(
                training_state=training_state,
                update_player=update_player,
                random_key=random_key,
                n=2 * metrics_period,
            )
        )
        pbar = tqdm(total=n_iterations, desc="Training", unit_scale=True)
        for k in range(n_loops):
            if k == 0:
                current_policy = training_state.avg_probs
                current_policy /= training_state.avg_probs.sum(axis=-1, keepdims=True)
                exp = self._exploitability_fn(policy_params=current_policy)
                pbar.set_postfix(exploitability=f"{exp:.1e}")

            random_key, subkey = jax.random.split(random_key)

            # Do n iterations
            training_state, update_player = _do_n_iterations(
                training_state, update_player, subkey
            )

            # Evaluate exploitability
            current_policy = training_state.avg_probs
            current_policy /= training_state.avg_probs.sum(axis=-1, keepdims=True)

            exp = self._exploitability_fn(policy_params=current_policy)
            pbar.set_postfix(exploitability=f"{exp:.1e}")
            pbar.update(metrics_period)

        return training_state
