from typing import Any, Literal
import math
import jax
import jax.numpy as jnp
from jaxtyping import (
    DTypeLike,
    Float,
    Array,
    ArrayLike,
    Inexact,
    PRNGKeyArray,
    Complex,
)

from . import util
from .convert import fft_frequencies, mel_frequencies, power_to_db


def stft(
    x: Float[ArrayLike, "*channels n_samples"],
    /,
    n_fft: int = 2048,
    hop_length: int | None = None,
    win_length: int | None = None,
    window: str | float | tuple | Float[ArrayLike, " win_length"] = "hann",
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

    win_dims = [1] * frames.ndim
    win_dims[-2] = len(fft_window)
    fft_window = fft_window.reshape(*win_dims)

    return jnp.fft.rfft(frames * fft_window, n=n_fft, axis=-2)


def istft(
    x: Complex[ArrayLike, "*channels n_freqs n_frames"],
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


def mel_filterbank(
    *,
    sr: float,
    n_fft: int,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: float | None = None,
    htk: bool = False,
    norm: Literal["slaney"] | float | None = "slaney",
    dtype: DTypeLike | None = None,
) -> Float[Array, " {n_mels} {n_fft}//2+1"]:
    if fmax is None:
        fmax = sr / 2

    fft_freqs = fft_frequencies(sr=sr, n_fft=n_fft).astype(dtype)
    mel_freqs = mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax, htk=htk).astype(dtype)
    fdiff = jnp.diff(mel_freqs)

    def _mel(i):
        lower = (-mel_freqs[i] + fft_freqs) / fdiff[i]
        upper = (mel_freqs[i + 2] - fft_freqs) / fdiff[i + 1]
        return jnp.maximum(0.0, jnp.minimum(lower, upper))

    mels = jax.vmap(_mel)(jnp.arange(n_mels))

    if norm == "slaney":
        enorm = 2.0 / (mel_freqs[2 : n_mels + 2] - mel_freqs[:n_mels])
        mels *= enorm[:, None]
    else:
        mels = util.normalize(mels, ord=norm, axis=-1)

    return mels


def cepstral_coefficients(
    S: Float[Array, "*channels n_freqs n_frames"],
    /,
    n_cc: int = 20,
    norm: str | None = "ortho",
    mag_scale: Literal["linear", "log", "db"] = "db",
    lifter: float = 0.0,
) -> Float[Array, "*channels {n_cc} n_frames"]:
    if mag_scale == "log":
        S = jnp.log(S + 1e-6)
    elif mag_scale == "db":
        S = power_to_db(S, amin=1e-6)

    M = jax.scipy.fft.dct(S, axis=-2, norm=norm)[..., :n_cc, :]

    if lifter > 0.0:
        li = jnp.sin(jnp.pi * jnp.arange(1, 1 + n_cc, dtype=S.dtype) / lifter)

        shape = [1] * M.ndim
        shape[-2] = n_cc
        M *= 1 + (lifter / 2) * li.reshape(shape)
    return M


def spectrogram(
    x: Float[ArrayLike, "*channels n_samples"],
    /,
    n_fft: int = 2048,
    hop_length: int | None = None,
    win_length: int | None = None,
    window: str | float | tuple | Float[ArrayLike, " win_length"] = "hann",
    center: bool = True,
    power: float | int = 2.0,
    pad_kwargs: dict[str, Any] = dict(),
) -> Inexact[Array, "*channels {n_fft}//2+1 n_frames"]:
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


def to_mel_scale(
    S: Float[Array, "*channels n_freqs n_frames"],
    /,
    sr: float,
    n_fft: int,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> Float[Array, "*channels {n_mels} n_frames"]:
    with jax.ensure_compile_time_eval():
        mels = mel_filterbank(
            sr=sr,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
        )

    return jnp.einsum("...fn,mf->...mn", S, mels)


def mel_spectrogram(
    x: Float[ArrayLike, "*channels n_samples"],
    /,
    sr: float,
    n_fft: int,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: float | None = None,
    hop_length: int | None = None,
    win_length: int | None = None,
    window: str | float | tuple | Float[ArrayLike, " win_length"] = "hann",
    center: bool = True,
    power: float | int = 2.0,
    pad_kwargs: dict[str, Any] = dict(),
) -> Float[Array, "*channels {n_mels} n_frames"]:
    S = spectrogram(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        power=power,
        pad_kwargs=pad_kwargs,
    )

    return to_mel_scale(
        S,
        sr=sr,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
    )


def mfcc(
    x: Float[ArrayLike, "*channels n_samples"],
    /,
    sr: float,
    n_fft: int,
    n_mfcc: int = 20,
    norm: str | None = "ortho",
    mag_scale: Literal["linear", "log", "db"] = "db",
    lifter: float = 0.0,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: float | None = None,
    hop_length: int | None = None,
    win_length: int | None = None,
    window: str | float | tuple | Float[ArrayLike, " win_length"] = "hann",
    center: bool = True,
    power: float | int = 2.0,
    pad_kwargs: dict[str, Any] = dict(),
) -> Float[Array, "*channels {n_mfcc} n_frames"]:
    S = mel_spectrogram(
        x,
        sr=sr,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        power=power,
        pad_kwargs=pad_kwargs,
    )

    return cepstral_coefficients(
        S,
        n_cc=n_mfcc,
        norm=norm,
        mag_scale=mag_scale,
        lifter=lifter,
    )


def griffin_lim(
    S: Float[ArrayLike, "*channels n_freqs n_frames"],
    /,
    key: PRNGKeyArray | None = None,
    n_iter: int = 32,
    n_fft: int | None = None,
    hop_length: int | None = None,
    win_length: int | None = None,
    window: str | float | tuple | Float[ArrayLike, " win_length"] = "hann",
    center: bool = True,
    length: int | None = None,
    momentum: float = 0.99,
    pad_kwargs: dict[str, Any] = dict(),
) -> Float[Array, "*channels n_samples"]:
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
