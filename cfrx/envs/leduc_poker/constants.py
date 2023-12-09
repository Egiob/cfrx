import os

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
