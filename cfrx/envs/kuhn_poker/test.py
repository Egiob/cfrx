# TEST 1
#  episode = Episode(
#     info_states=jnp.array([4, 9, 7, -1]),
#     actions=jnp.array([3, 1, 0, -1]),
#     rewards=jnp.array(
#         [
#             [0.0, 0.0],
#             [0.0, 0.0],
#             [0.0, 0.0],
#             [-2, 0],
#         ]
#     ),
#     action_masks=jnp.array(
#         [
#             [False, True, False, True],
#             [False, True, False, True],
#             [True, False, True, False],
#             [True, True, True, True],
#         ]
#     ),
#     current_players=jnp.array([0, 1, 0, -1]),
#     behavior_probs=jnp.array([0.5, 0.5, 0.5, 1.0]),
#     mask=jnp.array([1, 1, 1, 0]),
# )

# TEST 2
#     episode = Episode(
#     info_states=jnp.array([4, 6, -1, -1]),
#     actions=jnp.array([1, 0, -1, -1]),
#     rewards=jnp.array(
#         [
#             [0.0, 0.0],
#             [0.0, 0.0],
#             [2.0, 0.0],
#             [0.0, 0.0],
#         ]
#     ),
#     action_masks=jnp.array(
#         [
#             [False, True, False, True],
#             [True, False, True, False],
#             [True, True, True, True],
#             [True, True, True, True],
#         ]
#     ),
#     current_players=jnp.array([0, 1, -1, -1]),
#     behavior_probs=jnp.array([0.5, 0.5, 1.0, 1.0]),
#     mask=jnp.array([1, 1, 0, 0]),
# )
