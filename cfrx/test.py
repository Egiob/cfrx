import functools
from dataclasses import dataclass
from typing import NamedTuple

import beartype
import flax
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray, jaxtyped

from cfrx.algorithms.mccfr.outcome_sampling import MCCFRState
from cfrx.envs.kuhn_poker.constants import INFO_SETS
from cfrx.envs.kuhn_poker.env import KuhnPoker
from cfrx.policy import TabularPolicy

if __name__ == "__main__":

    @flax.struct.dataclass
    @jaxtyped(typechecker=beartype.beartype)
    class Data:
        a: Float[Array, "..."]

    @jaxtyped(typechecker=beartype.beartype)
    def f(x: Data) -> int:
        return 1

    data = Data(jnp.array([1, 2, 3]))

    jax.vmap(f)(data)  # type-check error
