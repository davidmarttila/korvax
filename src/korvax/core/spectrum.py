import math
from typing import Any
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array, ArrayLike, Complex
import equinox as eqx

from . import util


def stft(
    x: Float[Array, "*channels n_samples"],
    /,
    n_fft: int = 2048,
    hop_length: int | None = None,
    win_length: int | None = None,
    window: str | float | tuple | Float[ArrayLike, " win_length"] = "hann",
    center: bool = True,
    **pad_kwargs: Any,
) -> Complex[Array, "*channels {n_fft}//2+1 n_frames"]:
    """Compute the short-time Fourier transform (STFT) of a time-domain signal.

    Args:
        x: Input signal.
        n_fft: FFT size (number of samples per frame).
        hop_length: Hop (step) length between adjacent frames. If None, defaults to
            `win_length // 4`.
        win_length: Length of the analysis window. If None, defaults to `n_fft`.
            Ignored if `window` is an array.
        window: Either a 1d array containing the window to apply to each frame,
            or a window specification (see [get_window][korvax.util.get_window]).
        center: If True, pad the input so that frames are centered on their timestamps.
        **pad_kwargs: Additional keyword arguments forwarded to [pad_center][korvax.util.pad_center].

    Returns:
        STFT coefficients.
    """
    if win_length is None:
        win_length = n_fft

    if hop_length is None:
        hop_length = win_length // 4

    if center:
        x = util.pad_center(x, size=x.shape[-1] + n_fft, **pad_kwargs)

    frames = util.frame(x, frame_length=n_fft, hop_length=hop_length)

    if eqx.is_array(window):
        fft_window = jnp.asarray(window)
    else:
        fft_window = util.get_window(window, win_length, fftbins=True)  # pyright: ignore[reportArgumentType]

    win_dims = [1] * frames.ndim
    win_dims[-2] = len(fft_window)
    fft_window = fft_window.reshape(*win_dims)

    return jnp.fft.rfft(frames * fft_window, n=n_fft, axis=-2)


def istft(
    x: Complex[Array, "*channels n_freqs n_frames"],
    /,
    n_fft: int | None = None,
    hop_length: int | None = None,
    win_length: int | None = None,
    window: str | float | tuple | Float[ArrayLike, " win_length"] = "hann",
    center: bool = True,
    length: int | None = None,
) -> Float[Array, "*channels n_samples"]:
    """Compute the inverse short-time Fourier transform (ISTFT).

    Args:
        x: STFT coefficients.
        n_fft: FFT size (number of samples per frame).
        hop_length: Hop (step) length between adjacent frames. If None, defaults to
            `win_length // 4`.
        win_length: Length of the analysis window. If None, defaults to `n_fft`.
            Ignored if `window` is an array.
        window: Either a 1d array containing the window to apply to each frame,
            or a window specification (see [get_window][korvax.util.get_window]).
        center: If `True`, frames are assumed to be centered in time. If `False`, they
            are assumed to be left-aligned in time.
        length: If provided, the output will be trimmed or zero-padded to exactly this
            length.

    Returns:
        Reconstructed time-domain signal.
    """
    if n_fft is None:
        n_fft = (x.shape[-2] - 1) * 2

    if win_length is None:
        win_length = n_fft

    if hop_length is None:
        hop_length = win_length // 4

    if length:
        if center:
            padded_length = length + 2 * (n_fft // 2)
        else:
            padded_length = length
        n_frames = min(x.shape[-1], int(math.ceil(padded_length / hop_length)))
    else:
        n_frames = x.shape[-1]

    x = x[..., :n_frames]
    x = jnp.fft.irfft(x, n=n_fft, axis=-2)

    expected_length = n_fft + hop_length * (n_frames - 1)
    if length:
        expected_length = length
    elif center:
        expected_length -= n_fft

    with jax.ensure_compile_time_eval():
        if eqx.is_array(window):
            ifft_window = jnp.asarray(window)
        else:
            ifft_window = util.get_window(window, win_length, fftbins=True)  # pyright: ignore[reportArgumentType]

        ifft_window = util.pad_center(ifft_window, n_fft)

        win_dims = [1] * x.ndim
        win_dims[-2] = len(ifft_window)
        ifft_window = ifft_window.reshape(*win_dims)

        win_sumsq = (ifft_window / ifft_window.max()) ** 2
        win_sumsq = jnp.broadcast_to(win_sumsq, win_dims[:-1] + [x.shape[-1]])
        win_sumsq = util.overlap_and_add(win_sumsq, hop_length=hop_length)
        if center:
            win_sumsq = win_sumsq[..., n_fft // 2 :]
        win_sumsq = util.fix_length(win_sumsq, size=expected_length)
        win_sumsq = jnp.where(
            win_sumsq < jnp.finfo(win_sumsq.dtype).eps, 1.0, win_sumsq
        )

    x *= ifft_window

    x = util.overlap_and_add(x, hop_length=hop_length)
    if center:
        x = x[..., n_fft // 2 :]

    x = util.fix_length(x, size=expected_length)

    return x / win_sumsq
