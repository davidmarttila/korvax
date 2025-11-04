from collections.abc import Callable, Sequence
from typing import Literal

import jax.numpy as jnp

from jaxtyping import Float, Array, ArrayLike

from .. import stft


def spectral_convergence_loss(
    S_x: Float[Array, "*channels n_freq n_frames"],
    S_y: Float[Array, "*channels n_freq n_frames"],
    /,
    reduction: Literal["mean", "sum"] = "mean",
    eps: float = 1e-8,
) -> Float[Array, ""]:
    """
    Calculate the spectral convergence loss.

    Args:
        S_x (array): The magnitude spectrum of the first signal.
        S_y (array): The magnitude spectrum of the second signal.

    Returns:
        The spectral convergence loss.
    """
    numerator = jnp.linalg.norm(S_y - S_x, ord="fro", axis=(-2, -1))
    denominator = jnp.linalg.norm(S_y, ord="fro", axis=(-2, -1)) + eps
    loss = numerator / denominator
    if reduction == "mean":
        return jnp.mean(loss)
    elif reduction == "sum":
        return jnp.sum(loss)


def stft_magnitude_loss(
    S_x: Float[Array, "*channels n_freq n_frames"],
    S_y: Float[Array, "*channels n_freq n_frames"],
    /,
    log: bool = True,
    distance: Literal["L1", "L2"] = "L1",
    reduction: Literal["mean", "sum"] = "mean",
    log_fac: float = 1.0,
    log_eps: float = 1e-8,
) -> Float[Array, ""]:
    """
    Calculate the STFT magnitude loss.

    Log-magnitudes are calculated with `log(log_fac*x + log_eps)`, where `log_fac` controls the
    compression strength (larger value results in more compression), and `log_eps` can be used
    to control the range of the compressed output values (e.g., `log_eps>=1` ensures positive
    output values). The default values `log_fac=1` and `log_eps=0` correspond to plain log-compression.

    Args:
        x_mag (array): The magnitude spectrum of the first signal.
        y_mag (array): The magnitude spectrum of the second signal.
        log (bool): Whether to log-scale the STFT magnitudes.
        distance (str): Distance function ["L1", "L2"].
        reduction (str): Reduction of the loss elements ["mean", "sum", "none"].
        log_eps (float, optional): Constant value added to the magnitudes before evaluating the logarithm.
            Default: 1e-8
        log_fac (float, optional): Constant multiplication factor for the magnitudes before evaluating the logarithm.
            Default: 1.0
    Returns:
        The STFT magnitude loss.
    """
    if log:
        S_x = jnp.log(log_fac * S_x + log_eps)
        S_y = jnp.log(log_fac * S_y + log_eps)

    if distance == "L1":
        loss = jnp.abs(S_x - S_y)
    elif distance == "L2":
        loss = (S_x - S_y) ** 2

    if reduction == "mean":
        return jnp.mean(loss)
    elif reduction == "sum":
        return jnp.sum(loss)


def stft_loss(
    x: Float[Array, "*channels n_samples"],
    y: Float[Array, "*channels n_samples"],
    /,
    fft_size: int = 1024,
    hop_size: int = 256,
    win_length: int = 1024,
    window: str | float | tuple | Float[ArrayLike, " win_length"] = "hann",
    w_sc: float = 1.0,
    w_log_mag: float = 1.0,
    w_lin_mag: float = 0.0,
    w_phs: float = 0.0,
    filterbank: Float[Array, " n_filters {fft_size}//2+1"] | None = None,
    perceptual_weighting: Float[Array, " {fft_size}//2+1"] | None = None,
    scale_invariance: bool = False,
    eps: float = 1e-8,
    log_eps: float = 1e-8,
    log_fac: float = 1.0,
    reduction: Literal["mean", "sum"] = "mean",
    output: Literal["loss", "full"] = "loss",
    mag_distance: Literal["L1", "L2"] = "L1",
) -> Float[Array, ""] | tuple[Float[Array, ""], tuple[Float[Array, ""], ...]]:
    S_x = stft(
        x, n_fft=fft_size, hop_length=hop_size, win_length=win_length, window=window
    )
    S_y = stft(
        y, n_fft=fft_size, hop_length=hop_size, win_length=win_length, window=window
    )

    def broadcast_to_freq_axis(arr, axis=0):
        shape = [1] * S_x.ndim
        shape[-2] = arr.shape[axis]
        return arr.reshape(shape)

    if perceptual_weighting is not None:
        S_x = S_x * broadcast_to_freq_axis(perceptual_weighting)
        S_y = S_y * broadcast_to_freq_axis(perceptual_weighting)

    phs_x, phs_y, phs_loss = None, None, jnp.array(0.0, dtype=x.dtype)
    if w_phs:
        phs_x = jnp.angle(S_x)
        phs_y = jnp.angle(S_y)
        phs_loss = ((phs_x - phs_y) ** 2).mean()

    def mag(x):
        return jnp.sqrt((x * jnp.conj(x)).real.clip(min=eps))

    mag_x = mag(S_x)
    mag_y = mag(S_y)

    if filterbank is not None:
        mag_x = jnp.matmul(filterbank, mag_x)
        mag_y = jnp.matmul(filterbank, mag_y)

    if scale_invariance:
        alpha = (mag_x * mag_y).sum(axis=(-2, -1)) / (mag_y**2).sum(axis=(-2, -1))
        mag_y = mag_y * jnp.expand_dims(alpha, axis=-1)

    sc_mag_loss = (
        spectral_convergence_loss(mag_x, mag_y)
        if w_sc
        else jnp.array(0.0, dtype=x.dtype)
    )
    log_mag_loss = (
        stft_magnitude_loss(
            mag_x,
            mag_y,
            log=True,
            reduction=reduction,
            distance=mag_distance,
            log_fac=log_fac,
            log_eps=log_eps,
        )
        if w_log_mag
        else jnp.array(0.0, dtype=x.dtype)
    )
    lin_mag_loss = (
        stft_magnitude_loss(
            mag_x,
            mag_y,
            log=False,
            reduction=reduction,
            distance=mag_distance,
        )
        if w_lin_mag
        else jnp.array(0.0, dtype=x.dtype)
    )

    loss = (
        w_sc * sc_mag_loss
        + w_log_mag * log_mag_loss
        + w_lin_mag * lin_mag_loss
        + w_phs * phs_loss
    )

    if reduction == "sum":
        loss = loss.sum()
    elif reduction == "mean":
        loss = jnp.mean(loss)

    if output == "loss":
        return loss
    elif output == "full":
        return loss, (sc_mag_loss, log_mag_loss, lin_mag_loss, phs_loss)


def multi_resolution_stft_loss(
    x: Float[Array, "*channels n_samples"],
    y: Float[Array, "*channels n_samples"],
    fft_sizes: Sequence[int] = (1024, 2048, 512),
    hop_sizes: Sequence[int] = (120, 240, 50),
    win_lengths: Sequence[int] = (600, 1200, 240),
    window: str | float | tuple | Float[ArrayLike, " win_length"] = "hann",
    w_sc: float = 1.0,
    w_log_mag: float = 1.0,
    w_lin_mag: float = 0.0,
    w_phs: float = 0.0,
    make_filterbank: Callable[[int], Float[Array, "n_filters n_bins"]] | None = None,
    make_perceptual_weighting: Callable[[int], Float[Array, " n_bins"]] | None = None,
    scale_invariance: bool = False,
    eps: float = 1e-8,
    log_eps: float = 1e-8,
    log_fac: float = 1.0,
    reduction: Literal["mean", "sum"] = "mean",
    output: Literal["loss", "full"] = "loss",
    mag_distance: Literal["L1", "L2"] = "L1",
) -> (
    Float[Array, ""] | tuple[Float[Array, ""], tuple[tuple[Float[Array, ""], ...], ...]]
):
    outs = []

    for fs, hs, wl in zip(fft_sizes, hop_sizes, win_lengths):
        perceptual_weighting = None
        if make_perceptual_weighting is not None:
            perceptual_weighting = make_perceptual_weighting(fs)

        filterbank = None
        if make_filterbank is not None:
            filterbank = make_filterbank(fs)

        outs.append(
            stft_loss(
                x,
                y,
                fft_size=fs,
                hop_size=hs,
                win_length=wl,
                window=window,
                w_sc=w_sc,
                w_log_mag=w_log_mag,
                w_lin_mag=w_lin_mag,
                w_phs=w_phs,
                filterbank=filterbank,
                perceptual_weighting=perceptual_weighting,
                scale_invariance=scale_invariance,
                eps=eps,
                log_eps=log_eps,
                log_fac=log_fac,
                reduction=reduction,
                mag_distance=mag_distance,
                output=output,
            )
        )

    if output == "loss":
        return sum(outs, start=jnp.array(0.0, dtype=x.dtype)) / len(outs)
    elif output == "full":
        total_loss = jnp.array(0.0, dtype=x.dtype)
        sc_mag_loss, log_mag_loss, lin_mag_loss, phs_loss = [], [], [], []
        for out in outs:
            (
                loss,
                (
                    sc_loss,
                    log_loss,
                    lin_loss,
                    phs_l,
                ),
            ) = out  # type: ignore[unpack]
            total_loss += loss
            sc_mag_loss.append(sc_loss)
            log_mag_loss.append(log_loss)
            lin_mag_loss.append(lin_loss)
            phs_loss.append(phs_l)
        return total_loss / len(outs), (
            tuple(sc_mag_loss),
            tuple(log_mag_loss),
            tuple(lin_mag_loss),
            tuple(phs_loss),
        )
