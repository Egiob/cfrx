from __future__ import annotations

from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass

import pgx
from flax.struct import PyTreeNode
from jaxtyping import Array, Bool, Float, Int, PyTree, Shaped

# InfoState = PyTree
State = PyTree


class BaseEnv(ABC):
    @abstractmethod
    def action_to_string(cls, action: Int[Array, ""]) -> str:
        pass


# class State(ABC, PyTreeNode):
#     @abstractmethod
#     def update_info_state(
#         self, state: State, next_state: State, action: Int[Array, ""]
#     ) -> InfoState:
#         raise NotImplementedError

#     @abstractmethod
#     def info_state_to_str(self, info_state: InfoState) -> str:
#         raise NotImplementedError

#     @abstractmethod
#     def info_state_idx(self, info_state: InfoState) -> Int[Array, ""]:
#         raise NotImplementedError

#     chance_node: Bool[Array, "..."]
#     chance_prior: Float[Array, "..."]
#     info_state: Shaped[InfoState, "..."]

#     # @property
#     # @abstractmethod
#     # def chance_node(self) -> Bool[Array, ""]:
#     #     raise NotImplementedError

#     # @property
#     # @abstractmethod
#     # def chance_prior(self) -> Float[Array, "..."]:
#     #     raise NotImplementedError

#     # @property
#     # @abstractmethod
#     # def info_state(self) -> Shaped[InfoState, "..."]:
#     #     raise NotImplementedError


class Env(BaseEnv, pgx.v1.Env):
    pass
