from typing import Any, Literal
import jax
import jax.numpy as jnp
from jaxtyping import DTypeLike, Float, Array, ArrayLike

from . import spectrogram
from .. import util
from ..convert import fft_frequencies, mel_frequencies, power_to_db
from .._typing import _WindowSpec


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
    """Create a mel-scale filterbank.

    Args:
        sr: Sample rate of the audio signal.
        n_fft: FFT size (number of samples per frame).
        n_mels: Number of mel bands to generate.
        fmin: Minimum frequency (Hz).
        fmax: Maximum frequency (Hz). If None, defaults to `sr / 2`.
        htk: If True, use HTK formula for mel scale. Otherwise, use Slaney formula.
        norm: Normalization mode. If "slaney", use Slaney-style normalization.
            If a float, use L-norm normalization. If None, no normalization.
        dtype: Data type for the filterbank. If None, defaults to default float type.

    Returns:
        Mel filterbank matrix.
    """
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
    norm: Literal["backward", "ortho"] | None = "ortho",
    mag_scale: Literal["linear", "log", "db"] = "db",
    lifter: float = 0.0,
) -> Float[Array, "*channels {n_cc} n_frames"]:
    """Compute cepstral coefficients from a spectrogram via discrete cosine transform.

    Args:
        S: Input spectrogram.
        n_cc: Number of cepstral coefficients to return.
        norm: Normalization mode for DCT.
        mag_scale: Magnitude scaling to apply before DCT. Options are "linear" (no scaling),
            "log" (natural logarithm), or "db" (decibels).
        lifter: If greater than 0, apply liftering (cepstral filtering) with the specified
            coefficient.

    Returns:
        Cepstral coefficients.
    """
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


def to_mel_scale(
    S: Float[Array, "*channels n_freqs n_frames"],
    /,
    sr: float,
    n_fft: int,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> Float[Array, "*channels {n_mels} n_frames"]:
    """Convert a linear-frequency spectrogram to mel scale.

    Args:
        S: Input spectrogram.
        sr: Sample rate of the audio signal.
        n_fft: FFT size (number of samples per frame).
        n_mels: Number of mel bands to generate.
        fmin: Minimum frequency (Hz).
        fmax: Maximum frequency (Hz). If None, defaults to `sr / 2`.

    Returns:
        Mel-scale spectrogram.
    """
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
    window: _WindowSpec = "hann",
    center: bool = True,
    power: float | int = 2.0,
    pad_kwargs: dict[str, Any] = dict(),
) -> Float[Array, "*channels {n_mels} n_frames"]:
    """Compute a mel-scaled spectrogram from a time-domain signal.

    Args:
        x: Input signal.
        sr: Sample rate of the audio signal.
        n_fft: FFT size (number of samples per frame).
        n_mels: Number of mel bands to generate.
        fmin: Minimum frequency (Hz).
        fmax: Maximum frequency (Hz). If None, defaults to `sr / 2`.
        hop_length: Hop (step) length between adjacent frames. If None, defaults to
            `win_length // 4`.
        win_length: Length of the analysis window. If None, defaults to `n_fft`.
            Ignored if `window` is an array.
        window: Either a 1d array containing the window to apply to each frame,
            or a window specification (see [get_window][korvax.util.get_window]).
        center: If True, pad the input so that frames are centered on their timestamps.
        power: Exponent for the magnitude spectrogram. If 2.0, returns power spectrogram.
        pad_kwargs: Additional keyword arguments forwarded to [pad_center][korvax.util.pad_center].

    Returns:
        Mel-scale spectrogram.
    """
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
    norm: Literal["backward", "ortho"] | None = "ortho",
    mag_scale: Literal["linear", "log", "db"] = "db",
    lifter: float = 0.0,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: float | None = None,
    hop_length: int | None = None,
    win_length: int | None = None,
    window: _WindowSpec = "hann",
    center: bool = True,
    power: float | int = 2.0,
    pad_kwargs: dict[str, Any] = dict(),
) -> Float[Array, "*channels {n_mfcc} n_frames"]:
    """Compute mel-frequency cepstral coefficients (MFCCs) from a time-domain signal.

    Args:
        x: Input signal.
        sr: Sample rate of the audio signal.
        n_fft: FFT size (number of samples per frame).
        n_mfcc: Number of MFCCs to return.
        norm: Normalization mode for DCT.
        mag_scale: Magnitude scaling to apply before DCT. Options are "linear" (no scaling),
            "log" (natural logarithm), or "db" (decibels).
        lifter: If greater than 0, apply liftering (cepstral filtering) with the specified
            coefficient.
        n_mels: Number of mel bands to generate.
        fmin: Minimum frequency (Hz).
        fmax: Maximum frequency (Hz). If None, defaults to `sr / 2`.
        hop_length: Hop (step) length between adjacent frames. If None, defaults to
            `win_length // 4`.
        win_length: Length of the analysis window. If None, defaults to `n_fft`.
            Ignored if `window` is an array.
        window: Either a 1d array containing the window to apply to each frame,
            or a window specification (see [get_window][korvax.util.get_window]).
        center: If True, pad the input so that frames are centered on their timestamps.
        power: Exponent for the magnitude spectrogram. If 2.0, returns power spectrogram.
        pad_kwargs: Additional keyword arguments forwarded to [pad_center][korvax.util.pad_center].

    Returns:
        Mel-frequency cepstral coefficients.
    """
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
