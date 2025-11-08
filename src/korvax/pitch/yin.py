import math
from typing import Any

import jax
import jax.numpy as jnp

from jaxtyping import Array, ArrayLike, Bool, Float, Integer

from . import make_transition_matrix, viterbi_decode
from .. import util


def _cumulative_mean_normalized_difference(
    x: Float[Array, "*channels frame_length n_frames"],
    min_period: int,
    max_period: int,
) -> Float[Array, "*channels {max_period}-{min_period}+1 n_frames"]:
    """Cumulative mean normalized difference function (equation 8 in [#]_)

    .. [#] De Cheveigné, Alain, and Hideki Kawahara.
        "YIN, a fundamental frequency estimator for speech and music."
        The Journal of the Acoustical Society of America 111.4 (2002): 1917-1930.

    Parameters
    ----------
    y_frames : np.ndarray [shape=(frame_length, n_frames)]
        framed audio time series.
    min_period : int > 0 [scalar]
        minimum period.
    max_period : int > 0 [scalar]
        maximum period.

    Returns
    -------
    yin_frames : np.ndarray [shape=(max_period-min_period+1,n_frames)]
        Cumulative mean normalized difference function for each frame.
    """
    acf_frames = util.autocorrelate(x, max_size=max_period + 1, axis=-2)

    # Energy terms.
    yin_frames = jnp.square(x)
    yin_frames = jnp.cumsum(yin_frames, axis=-2)

    # Difference function: d(k) = 2 * (ACF(0) - ACF(k)) - sum_{m=0}^{k-1} y(m)^2
    k = slice(1, max_period + 1)
    yin_frames = yin_frames.at[..., 0, :].set(0.0)
    yin_frames = yin_frames.at[..., k, :].set(
        2 * (acf_frames[..., 0:1, :] - acf_frames[..., k, :])
        - yin_frames[..., : k.stop - 1, :]
    )

    # Cumulative mean normalized difference function.
    yin_numerator = yin_frames[..., min_period : max_period + 1, :]
    # broadcast this shape to have leading ones
    k_range = util.expand_to(jnp.r_[k], ndim=yin_frames.ndim, axes=-2)

    cumulative_mean = jnp.cumsum(yin_frames[..., k, :], axis=-2) / k_range
    yin_denominator = cumulative_mean[..., min_period - 1 : max_period, :]
    yin_frames: jnp.ndarray = yin_numerator / (
        yin_denominator + util.feps(yin_denominator)
    )
    return yin_frames


def yin(
    x: Float[ArrayLike, "*channels n_samples"],
    /,
    fmin: float,
    fmax: float,
    sr: float,
    frame_length: int = 2048,
    hop_length: int | None = None,
    trough_threshold: float = 0.1,
    center: bool = True,
    pad_kwargs: dict[str, Any] = dict(),
) -> Float[Array, "*channels n_frames"]:
    x = jnp.asarray(x)

    # Set the default hop if it is not already specified.
    if hop_length is None:
        hop_length = frame_length // 4

    # Pad the time series so that frames are centered
    if center:
        x = util.pad_center(
            x,
            size=x.shape[-1] + frame_length,
            pad_kwargs=pad_kwargs,
        )

    # Frame audio.
    frames = util.frame(x, frame_length=frame_length, hop_length=hop_length)

    # Calculate minimum and maximum periods
    min_period = int(math.floor(sr / fmax))
    max_period = min(int(math.ceil(sr / fmin)), frame_length - 1)

    # Calculate cumulative mean normalized difference function.
    yin_frames = _cumulative_mean_normalized_difference(frames, min_period, max_period)

    parabolic_shifts = util.parabolic_peak_shifts(yin_frames, axis=-2)

    # Find local minima.
    is_trough = util.localmin(yin_frames, axis=-2)
    is_trough = is_trough.at[..., 0, :].set(
        yin_frames[..., 0, :] < yin_frames[..., 1, :]
    )

    # Find minima below peak threshold.
    is_threshold_trough = jnp.logical_and(is_trough, yin_frames < trough_threshold)

    # Absolute threshold.
    # "The solution we propose is to set an absolute threshold and choose the
    # smallest value of tau that gives a minimum of d' deeper than
    # this threshold. If none is found, the global minimum is chosen instead."

    global_min = jnp.argmin(yin_frames, axis=-2, keepdims=True)
    yin_period = jnp.argmax(is_threshold_trough, axis=-2, keepdims=True)

    no_trough_below_threshold = jnp.all(~is_threshold_trough, axis=-2, keepdims=True)
    yin_period = jnp.where(no_trough_below_threshold, global_min, yin_period)

    # Refine peak by parabolic interpolation.

    yin_period = (
        min_period
        + yin_period
        + jnp.take_along_axis(parabolic_shifts, yin_period, axis=-2)
    )[..., 0, :]

    # Convert period to fundamental frequency.
    f0: jnp.ndarray = sr / yin_period
    return f0


def pyin_emission_probabilities(
    x: Float[ArrayLike, "*channels n_samples"],
    /,
    fmin: float,
    fmax: float,
    sr: float,
    frame_length: int = 2048,
    hop_length: int | None = None,
    n_thresholds: int = 100,
    beta_parameters: tuple[float, float] = (2, 18),
    boltzmann_parameter: float = 2,
    resolution: float = 0.1,
    no_trough_prob: float = 0.01,
    normalize: bool = False,
    center: bool = True,
    pad_kwargs: dict[str, Any] = dict(),
) -> Float[Array, "*channels n_bins n_frames"]:
    x = jnp.asarray(x)
    if hop_length is None:
        hop_length = frame_length // 4

    if center:
        x = util.pad_center(
            x,
            size=x.shape[-1] + frame_length,
            pad_kwargs=pad_kwargs,
        )

    frames = util.frame(x, frame_length=frame_length, hop_length=hop_length)

    # Calculate minimum and maximum periods
    min_period = int(math.floor(sr / fmax))
    max_period = min(int(math.ceil(sr / fmin)), frame_length - 1)

    # Calculate cumulative mean normalized difference function.
    yin_frames = _cumulative_mean_normalized_difference(frames, min_period, max_period)

    n_periods = yin_frames.shape[-2]

    parabolic_shifts = util.parabolic_peak_shifts(yin_frames, axis=-2)

    thresholds = jnp.linspace(0, 1, n_thresholds)
    beta_cdf = jax.scipy.stats.beta.cdf(
        thresholds, beta_parameters[0], beta_parameters[1]
    )
    beta_probs = jnp.diff(beta_cdf)

    n_bins_per_semitone = int(jnp.ceil(1.0 / resolution))
    n_pitch_bins = int(jnp.floor(n_bins_per_semitone * 12 * jnp.log2(fmax / fmin))) + 1

    def _boltzmann_pmf(
        k: Integer[Array, " n_periods n_thresholds"],
        N: Integer[Array, " n_thresholds"],
    ) -> Float[Array, "n_periods n_thresholds"]:
        e_lmbda = jnp.exp(-boltzmann_parameter)
        e_k = jnp.exp(-boltzmann_parameter * k)
        e_N = jnp.exp(-boltzmann_parameter * N)[None, :]
        return (1 - e_lmbda) / (1 - e_N + util.feps(e_N)) * e_k

    def _pyin_helper(
        yin_frame: Float[Array, " n_periods"],
        para_shifts: Float[Array, " n_periods"],
    ) -> Float[Array, " n_pitch_bins"]:
        is_trough = util.localmin(yin_frame, axis=-1)
        is_trough = is_trough.at[0].set(yin_frame[0] < yin_frame[1])

        trough_positions = jnp.logical_and(
            is_trough[:, None], jnp.less(yin_frame[:, None], thresholds[None, 1:])
        )

        n_troughs = trough_positions.sum(axis=0)
        jax.debug.print("{n_troughs}", n_troughs=n_troughs)

        trough_prior = jnp.where(
            trough_positions,
            _boltzmann_pmf(k=jnp.cumsum(trough_positions, axis=0), N=n_troughs),
            0.0,
        )

        probs = jnp.einsum("pt,t->p", trough_prior, beta_probs)

        global_min = jnp.argmin(yin_frame)

        n_thresholds_without_trough_below = jnp.count_nonzero(n_troughs == 0)
        jax.debug.print(
            "{n_thresholds_without_trough_below}",
            n_thresholds_without_trough_below=n_thresholds_without_trough_below,
        )
        probs = probs.at[global_min].add(
            no_trough_prob * beta_cdf[n_thresholds_without_trough_below]
        )

        period_candidates = min_period + jnp.arange(n_periods)
        period_candidates = period_candidates + para_shifts
        f0_candidates = sr / period_candidates

        f0_bins = (
            jnp.round(n_bins_per_semitone * 12 * jnp.log2(f0_candidates / fmin))
            .astype(jnp.int32)
            .clip(0, n_pitch_bins - 1)
        )

        # todo also interpolate peak **height**? librosa doesn't

        observation_probs = jnp.zeros((n_pitch_bins,))
        observation_probs = observation_probs.at[f0_bins].add(probs)

        return observation_probs

    in_shape = yin_frames.shape
    yin_frames_flat = yin_frames.swapaxes(-2, -1).reshape(-1, n_periods)
    parabolic_shifts_flat = parabolic_shifts.swapaxes(-2, -1).reshape(-1, n_periods)
    yin_probs = jax.vmap(_pyin_helper)(yin_frames_flat, parabolic_shifts_flat)
    yin_probs = yin_probs.reshape(
        in_shape[:-2] + (in_shape[-1], n_pitch_bins)
    ).swapaxes(-2, -1)

    if normalize:
        yin_probs = yin_probs / (
            jnp.sum(yin_probs, axis=-2, keepdims=True) + util.feps(yin_probs)
        )

    return yin_probs


def pyin(
    x: Float[ArrayLike, "*channels n_samples"],
    /,
    fmin: float,
    fmax: float,
    sr: float,
    frame_length: int = 2048,
    hop_length: int | None = None,
    n_thresholds: int = 100,
    beta_parameters: tuple[float, float] = (2.0, 18.0),
    boltzmann_parameter: float = 2.0,
    resolution: float = 0.1,
    no_trough_prob: float = 0.01,
    max_transition_rate: float = 35.92,
    switch_prob: float = 0.01,
    fill_na: float | None = jnp.nan,
    center: bool = True,
    pad_kwargs: dict[str, Any] = dict(),
) -> tuple[
    Float[Array, "*channels n_frames"],
    Bool[Array, "*channels n_frames"],
    Float[Array, "*channels n_frames"],
]:
    if hop_length is None:
        hop_length = frame_length // 4

    emission_probs = pyin_emission_probabilities(
        x,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        frame_length=frame_length,
        hop_length=hop_length,
        n_thresholds=n_thresholds,
        beta_parameters=beta_parameters,
        boltzmann_parameter=boltzmann_parameter,
        resolution=resolution,
        no_trough_prob=no_trough_prob,
        normalize=False,
        center=center,
        pad_kwargs=pad_kwargs,
    )

    n_pitch_bins = emission_probs.shape[-2]
    voiced_probs = jnp.sum(emission_probs, axis=-2, keepdims=True).clip(0.0, 1.0)

    unvoiced_probs = (1 - voiced_probs) / n_pitch_bins
    unvoiced_probs = jnp.broadcast_to(unvoiced_probs, emission_probs.shape)

    emission_probs = jnp.concat([emission_probs, unvoiced_probs], axis=-2)

    bins_per_semitone = int(jnp.ceil(1.0 / resolution))

    transition_matrix = make_transition_matrix(
        max_transition_rate,
        hop_length,
        sr,
        bins_per_semitone,
        n_pitch_bins,
    )

    switch_matrix = jnp.array(
        ((1 - switch_prob, switch_prob), (switch_prob, 1 - switch_prob))
    )

    transition_matrix = jnp.kron(switch_matrix, transition_matrix)

    p_init = jnp.full((2 * n_pitch_bins,), 1 / (2 * n_pitch_bins))

    states = viterbi_decode(
        p_init,
        transition_matrix,
        jnp.log(emission_probs + util.feps(emission_probs)),
    )

    freqs = fmin * 2 ** (jnp.arange(n_pitch_bins) / (12 * bins_per_semitone))
    f0 = freqs[states % n_pitch_bins]
    voiced_flag = states < n_pitch_bins
    if fill_na is not None:
        f0 = jnp.where(voiced_flag, f0, fill_na)

    return f0, voiced_flag, voiced_probs.squeeze(axis=-2)
