from __future__ import annotations

from abc import ABC, abstractmethod

from jaxtyping import Array, Bool, Float, Int


class InfoState(ABC):
    pass


class Env(ABC):
    @abstractmethod
    def action_to_string(cls, action: Int[Array, ""]) -> str:
        pass


class State(ABC):
    @abstractmethod
    def update_info_state(
        self, state: State, next_state: State, action: Int[Array, ""]
    ) -> InfoState:
        pass

    @abstractmethod
    def info_state_to_str(self, info_state: InfoState) -> str:
        pass

    @abstractmethod
    def info_state_idx(self, info_state: InfoState) -> Int[Array, ""]:
        pass

    @property
    @abstractmethod
    def chance_node(self) -> Bool[Array, ""]:
        pass

    @property
    @abstractmethod
    def chance_prior(self) -> Float[Array, "..."]:
        pass

    @property
    @abstractmethod
    def legal_action_mask(self) -> Bool[Array, "..."]:
        pass
