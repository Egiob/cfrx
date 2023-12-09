import os

import jax
import numpy as np

current_path = os.path.dirname(os.path.abspath(__file__))

INFO_SETS = dict(
    np.load(
        os.path.join(
            current_path,
            "data",
            "info_states.npz",
        )
    )
)

INFO_SET_ACTION_MASK = {
    "0": [False, True, False, True],
    "0p": [False, True, False, True],
    "0b": [True, False, True, False],
    "0pb": [True, False, True, False],
    "1": [False, True, False, True],
    "1p": [False, True, False, True],
    "1b": [True, False, True, False],
    "1pb": [True, False, True, False],
    "2": [False, True, False, True],
    "2p": [False, True, False, True],
    "2b": [True, False, True, False],
    "2pb": [True, False, True, False],
}


def get_kuhn_optimal_policy(alpha: float) -> dict:
    assert 0 <= alpha <= 1 / 3
    optimal_probs = {
        "0": [0.0, alpha, 0.0, 1 - alpha],
        "0p": [0.0, 1 / 3, 0.0, 2 / 3],
        "0b": [0.0, 0.0, 1.0, 0.0],
        "0pb": [0.0, 0.0, 1.0, 0.0],
        "1": [0.0, 0.0, 0.0, 1.0],
        "1p": [0.0, 0.0, 0.0, 1.0],
        "1b": [1 / 3, 0.0, 2 / 3, 0.0],
        "1pb": [alpha + 1 / 3, 0.0, 2 / 3 - alpha, 0.0],
        "2": [0.0, 3 * alpha, 0, 1 - 3 * alpha],
        "2p": [0.0, 1.0, 0.0, 0.0],
        "2b": [1.0, 0.0, 0.0, 0.0],
        "2pb": [1.0, 0.0, 0.0, 0.0],
    }
    return optimal_probs


KUHN_UNIFORM_POLICY = jax.tree_map(
    lambda x: x / x.sum(), {k: np.array(x) for k, x in INFO_SET_ACTION_MASK.items()}
)
