from typing import NamedTuple

from jaxtyping import Array, Bool, Float, Int

from cfrx.envs.base import InfoState


class Episode(NamedTuple):
    info_state: InfoState
    action: Float[Array, "..."]
    reward: Float[Array, "..."]
    action_mask: Bool[Array, "..."]
    current_player: Int[Array, "..."]
    behavior_prob: Float[Array, "..."]
    mask: Bool[Array, "..."]
    chance_node: Bool[Array, "..."]
