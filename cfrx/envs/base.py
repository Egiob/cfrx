from __future__ import annotations

from abc import ABC, abstractmethod

import pgx.core
from jaxtyping import Array, Int, PyTree

InfoState = PyTree
State = PyTree


class BaseEnv(ABC):
    @abstractmethod
    def action_to_string(cls, action: Int[Array, ""]) -> str:
        pass


class Env(BaseEnv, pgx.core.Env):
    pass
