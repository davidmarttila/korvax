# from dynamax.hidden_markov_model import hmm_posterior_mode


# def make_transition_matrix(
#     max_transition_rate: float,
#     hop_length: int,
#     sr: float,
#     bins_per_semitone: int,
#     n_pitch_bins: int,
# ) -> Float[Array, "{n_pitch_bins} {n_pitch_bins}"]:
#     max_semitones_per_frame = round(max_transition_rate * 12 * hop_length / sr)
#     transition_width = max_semitones_per_frame * bins_per_semitone + 1

#     def _transition_row(i):
#         row = get_window("triang", transition_width)
#         row = row / jnp.sum(row)

#         row = jnp.pad(row, (0, n_pitch_bins), mode="constant")
#         row = jnp.roll(row, shift=i)

#         cut_right = transition_width // 2
#         if transition_width % 2 == 1:
#             cut_right += 1
#         return row[transition_width // 2 : -cut_right]

#     return jax.vmap(_transition_row)(jnp.arange(n_pitch_bins))


# def viterbi_decode(
#     initial_distribution: Float[Array, " n_states"],
#     transition_matrix: Float[Array, " n_states n_states"],
#     log_likelihoods: Float[Array, "*channels n_states n_frames"],
# ) -> Integer[Array, "*channels n_frames"]:
#     in_shape = log_likelihoods.shape
#     flat_lls = log_likelihoods.reshape((-1, in_shape[-2], in_shape[-1]))

#     states = jax.vmap(hmm_posterior_mode, in_axes=(None, None, 0))(  # pyright: ignore[reportPossiblyUnboundVariable]
#         initial_distribution, transition_matrix, flat_lls.swapaxes(-2, -1)
#     )

#     return states.reshape(in_shape[:-2] + (-1,))
