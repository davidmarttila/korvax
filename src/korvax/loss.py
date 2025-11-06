from collections.abc import Callable, Sequence
from typing import Literal
from functools import partial

import jax
import jax.numpy as jnp

from jaxtyping import Float, Inexact, Array, ArrayLike

from . import util, spectrogram


TransformFn = Callable[
    [Float[Array, "*channels n_samples"]], Inexact[Array, "*channels n_bins n_frames"]
]

ScaleFn = Callable[
    [Inexact[Array, "*channels n_bins_in n_frames"]],
    Inexact[Array, "*channels n_bins_out n_frames"],
]

LossFn = Callable[
    [
        Inexact[Array, "*channels n_bins n_frames"],
        Inexact[Array, "*channels n_bins n_frames"],
    ],
    Float[Array, ""],
]


def spectral_convergence_loss(
    S_x: Float[Array, "*channels n_freq n_frames"],
    S_y: Float[Array, "*channels n_freq n_frames"],
) -> Float[Array, ""]:
    numerator = jnp.linalg.norm(S_y - S_x, ord="fro", axis=(-2, -1))
    denominator = jnp.linalg.norm(S_y, ord="fro", axis=(-2, -1))
    loss = numerator / (denominator + util.feps(denominator))
    return jnp.mean(loss)


def elementwise_loss(
    x: Float[Array, "*dims"],
    y: Float[Array, "*dims"],
    /,
    metric: Literal["L1", "L2"] = "L1",
) -> Float[Array, ""]:
    if metric == "L1":
        loss = jnp.abs(x - y)
    elif metric == "L2":
        loss = (x - y) ** 2

    return jnp.mean(loss)


def _quantile_function(qs, cws, xs):
    n = xs.shape[0]
    cws = cws.T
    qs = qs.T
    idx = jnp.searchsorted(cws, qs).T
    return jnp.take_along_axis(xs, jnp.clip(idx, 0, n - 1), axis=0)


def _wasserstein_1d(S_x, S_y, positions, p=1, limit_quantile_range=False):
    u_cumweights = jnp.cumsum(S_x, 0)
    v_cumweights = jnp.cumsum(S_y, 0)

    qs = jnp.sort(jnp.concatenate((u_cumweights, v_cumweights), 0), 0)
    u_quantiles = _quantile_function(qs, u_cumweights, positions)
    v_quantiles = _quantile_function(qs, v_cumweights, positions)
    qs = jnp.pad(qs, pad_width=[(1, 0)] + (qs.ndim - 1) * [(0, 0)], mode="constant")
    delta = qs[1:, ...] - qs[:-1, ...]

    if limit_quantile_range:
        delta = jnp.where(qs[..., 1:] > 1, jnp.zeros_like(delta), delta)

    diff_quantiles = jnp.abs(u_quantiles - v_quantiles)

    if p == 1:
        return jnp.sum(delta * diff_quantiles, axis=0)
    return jnp.sum(delta * jnp.power(diff_quantiles, p), axis=0)


def spectral_optimal_transport_loss(
    S_x: Float[Array, "*channels n_bins n_frames"],
    S_y: Float[Array, "*channels n_bins n_frames"],
    /,
    positions: Float[Array, " n_bins"] | None = None,
    p: int = 2,
    normalize: bool = True,
    balanced: bool = True,
    limit_quantile_range: bool = False,
) -> Float[Array, ""]:
    n_bins = S_x.shape[-2]

    if positions is None:
        positions = jnp.linspace(0, 1, num=n_bins, endpoint=False, dtype=S_x.dtype)

    S_x = S_x.swapaxes(-1, -2).reshape(-1, n_bins)
    S_y = S_y.swapaxes(-1, -2).reshape(-1, n_bins)

    total_mass_x = jnp.sum(S_x, axis=-1, keepdims=True) + util.feps(S_x)
    total_mass_y = jnp.sum(S_y, axis=-1, keepdims=True) + util.feps(S_y)

    if normalize:
        S_x = S_x / total_mass_x
        if balanced:
            S_y = S_y / total_mass_y
        else:
            S_y = S_y / total_mass_x
    elif balanced:
        S_y = S_y * (total_mass_x / total_mass_y)

    return jax.vmap(
        _wasserstein_1d,
        in_axes=(0, 0, None, None, None),
    )(S_x, S_y, positions, p, limit_quantile_range).mean()


def time_frequency_loss(
    x: Float[Array, "*channels n_samples"],
    y: Float[Array, "*channels n_samples"],
    /,
    transform_fn: TransformFn,
    loss_fn: LossFn | Sequence[LossFn],
    scale_fn: ScaleFn | Sequence[ScaleFn] | None = None,
    weights: Sequence[float] | Float[ArrayLike, " n_losses"] | None = None,
) -> Float[Array, ""]:
    if not isinstance(loss_fn, Sequence):
        loss_fn = [loss_fn]

    if scale_fn is None:
        scale_fn = [lambda S: S] * len(loss_fn)

    if not isinstance(scale_fn, Sequence):
        scale_fn = [scale_fn]

    assert (n_losses := len(scale_fn)) == len(loss_fn)

    if weights is None:
        weights = jnp.ones(len(loss_fn), dtype=x.dtype) / n_losses
    else:
        weights = jnp.array(weights, dtype=x.dtype)

    assert len(weights) == n_losses

    loss_total = jnp.array(0.0, dtype=x.dtype)

    x = transform_fn(x)
    y = transform_fn(y)

    for i in range(n_losses):
        loss = loss_fn[i](scale_fn[i](x), scale_fn[i](y))
        loss_total += weights[i] * loss

    return loss_total


def mrstft_loss(
    x: Float[Array, "*channels n_samples"],
    y: Float[Array, "*channels n_samples"],
    /,
    hop_lengths: Sequence[int] = (32, 64, 128, 256, 512, 1024),
    win_lengths: Sequence[int] = (64, 128, 256, 512, 1024, 2048),
    fft_sizes: Sequence[int] | None = None,
    window: str | float | tuple = "hann",
    w_lin: float = 1.0,
    w_log: float = 1.0,
    lin_dist: Literal["L1", "L2"] = "L1",
    log_dist: Literal["L1", "L2"] = "L1",
    log_fac: float = 1.0,
    log_eps: float = 1e-7,
) -> Float[Array, ""]:
    if fft_sizes is None:
        fft_sizes = win_lengths

    assert (n_res := len(win_lengths)) == len(hop_lengths) == len(fft_sizes)

    def log_scale(S):
        return jnp.log(log_fac * S + log_eps)

    scale_fns = [lambda S: S, log_scale]

    loss_fns = [
        partial(elementwise_loss, metric=lin_dist),
        partial(elementwise_loss, metric=log_dist),
    ]

    weights = [w_lin, w_log]

    loss_total = jnp.array(0.0, dtype=x.dtype)

    for i in range(n_res):
        transform_fn = partial(
            spectrogram,
            win_length=win_lengths[i],
            hop_length=hop_lengths[i],
            n_fft=fft_sizes[i],
            window=window,
            power=1,
        )

        loss_total += time_frequency_loss(
            x,
            y,
            transform_fn=transform_fn,
            scale_fn=scale_fns,
            loss_fn=loss_fns,
            weights=weights,
        )

    return loss_total / n_res


def smooth_mrstft_loss(
    x: Float[Array, "*channels n_samples"],
    y: Float[Array, "*channels n_samples"],
) -> Float[Array, ""]:
    return mrstft_loss(
        x,
        y,
        hop_lengths=(32, 63, 128, 254, 510, 1026),
        win_lengths=(67, 127, 257, 509, 1021, 2053),
        window="flattop",
        w_lin=0.0,
        w_log=1.0,
        log_dist="L2",
        log_fac=1.0,
        log_eps=1.0,
    )
