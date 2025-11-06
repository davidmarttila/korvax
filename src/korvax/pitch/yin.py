import math
from typing import Any

import jax.numpy as jnp

from jaxtyping import Array, ArrayLike, Float

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
    sr: float = 22050,
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
