from typing import Any
import math
import jax
import jax.numpy as jnp
from jaxtyping import (
    Float,
    Array,
    ArrayLike,
    Inexact,
    PRNGKeyArray,
    Complex,
)

from .. import util
from .._typing import _WindowSpec


def stft(
    x: Float[ArrayLike, "*channels n_samples"],
    /,
    n_fft: int = 2048,
    hop_length: int | None = None,
    win_length: int | None = None,
    window: _WindowSpec = "hann",
    center: bool = True,
    pad_kwargs: dict[str, Any] = dict(),
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

    x = jnp.asarray(x)

    if center:
        x = util.pad_center(x, size=x.shape[-1] + n_fft, pad_kwargs=pad_kwargs)

    frames = util.frame(x, frame_length=n_fft, hop_length=hop_length)

    fft_window = util.get_window(
        window,
        win_length,
        fftbins=True,
        dtype=frames.dtype,
    )

    if len(fft_window) < n_fft:
        fft_window = util.pad_center(fft_window, n_fft)

    fft_window = util.expand_to(fft_window, frames.ndim, -2)

    return jnp.fft.rfft(frames * fft_window, n=n_fft, axis=-2)


def istft(
    x: Complex[ArrayLike, "*channels n_freqs n_frames"],
    /,
    n_fft: int | None = None,
    hop_length: int | None = None,
    win_length: int | None = None,
    window: _WindowSpec = "hann",
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
    x = jnp.asarray(x)

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
        ifft_window = util.get_window(
            window,
            win_length,
            fftbins=True,
            dtype=x.dtype,
        )

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


def spectrogram(
    x: Float[ArrayLike, "*channels n_samples"],
    /,
    n_fft: int = 2048,
    hop_length: int | None = None,
    win_length: int | None = None,
    window: _WindowSpec = "hann",
    center: bool = True,
    power: float | int | None = 2.0,
    pad_kwargs: dict[str, Any] = dict(),
) -> Inexact[Array, "*channels {n_fft}//2+1 n_frames"]:
    """Compute the magnitude spectrogram of a time-domain signal.

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
        power: Exponent for the magnitude spectrogram. If 2.0, returns power spectrogram.
            If None, returns complex STFT coefficients.
        pad_kwargs: Additional keyword arguments forwarded to [pad_center][korvax.util.pad_center].

    Returns:
        Magnitude spectrogram.
    """
    x = stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_kwargs=pad_kwargs,
    )

    if power is None:
        return x

    x = x * jnp.conj(x)
    return x.real if power == 2 else x.real ** (power / 2)


def griffin_lim(
    S: Float[ArrayLike, "*channels n_freqs n_frames"],
    /,
    key: PRNGKeyArray | None = None,
    n_iter: int = 32,
    n_fft: int | None = None,
    hop_length: int | None = None,
    win_length: int | None = None,
    window: _WindowSpec = "hann",
    center: bool = True,
    length: int | None = None,
    momentum: float = 0.99,
    pad_kwargs: dict[str, Any] = dict(),
) -> Float[Array, "*channels n_samples"]:
    """Reconstruct a time-domain signal from a magnitude spectrogram using the Griffin-Lim algorithm.

    Args:
        S: Magnitude spectrogram.
        key: JAX PRNG key for random phase initialization. If None, uses zero phase
            initialization.
        n_iter: Number of Griffin-Lim iterations to perform.
        n_fft: FFT size (number of samples per frame). If None, inferred from spectrogram
            shape.
        hop_length: Hop (step) length between adjacent frames. If None, defaults to
            `win_length // 4`.
        win_length: Length of the analysis window. If None, defaults to `n_fft`.
            Ignored if `window` is an array.
        window: Either a 1d array containing the window to apply to each frame,
            or a window specification (see [get_window][korvax.util.get_window]).
        center: If True, frames are assumed to be centered in time. If False, they
            are assumed to be left-aligned in time.
        length: If provided, the output will be trimmed or zero-padded to exactly this
            length.
        momentum: Momentum parameter for fast Griffin-Lim (typically between 0 and 1).
        pad_kwargs: Additional keyword arguments forwarded to [pad_center][korvax.util.pad_center].

    Returns:
        Reconstructed time-domain signal.
    """
    S = jnp.asarray(S)

    if n_fft is None:
        n_fft = (S.shape[-2] - 1) * 2

    complex_dtype = jnp.result_type(S.dtype, 1j)

    if key is None:
        angles = S.astype(complex_dtype)
    else:
        angles = jax.random.uniform(
            key, S.shape, minval=0.0, maxval=2 * jnp.pi, dtype=S.dtype
        )
        angles = jnp.cos(angles) + 1j * jnp.sin(angles)
        angles *= S

    def step(carry, _):
        prev_rebuilt, angles = carry

        inverse = istft(
            angles,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
        )
        rebuilt = stft(
            inverse,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            pad_kwargs=pad_kwargs,
        )

        angles = rebuilt
        angles -= (momentum / (1 + momentum)) * prev_rebuilt
        angles /= jnp.abs(angles) + util.feps(angles)
        angles *= S
        return (rebuilt, angles), None

    (_, angles), _ = jax.lax.scan(
        step, init=(jnp.zeros_like(angles), angles), length=n_iter
    )

    return istft(
        angles,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        length=length,
    )
