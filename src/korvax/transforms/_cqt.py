from typing import Literal
import jax
import jax.numpy as jnp
import jax.lax as lax
from jaxtyping import Array, Complex, DTypeLike, Float, Inexact, Integer

from .. import util


def create_vqt_kernels(
    Q: float,
    sr: float,
    fmin: float,
    n_bins: int | None = 84,
    bins_per_octave: int = 12,
    norm: float | int = 1,
    window: str | float | tuple = "hann",
    fmax: float | None = None,
    topbin_check: bool = True,
    gamma: float = 0.0,
    dtype: DTypeLike | None = None,
) -> tuple[
    Complex[Array, " n_bins fft_len"],
    Integer[Array, " n_bins"],
    Float[Array, " n_bins"],
]:
    """Create Variable-Q Transform (VQT) filter kernels.

    Args:
        Q: Q-factor for filter bandwidth. Higher values give finer frequency resolution.
        sr: Sample rate of the input signal.
        fmin: Minimum frequency (Hz).
        n_bins: Number of frequency bins. Ignored if `fmax` is provided.
        bins_per_octave: Number of bins per octave.
        norm: Normalization mode for the filters (p-norm to use).
        window: Window specification (see [get_window][korvax.util.get_window]).
        fmax: Maximum frequency (Hz). If provided, `n_bins` is calculated automatically.
        topbin_check: If True, raise an error if the highest frequency exceeds Nyquist.
        gamma: Bandwidth offset parameter for variable-Q transform.
        dtype: Data type for frequency array. If None, uses default dtype.

    Returns:
        Tuple of (kernels, lengths, freqs) where kernels are the complex filter banks,
        lengths are the individual filter lengths, and freqs are the center frequencies.
    """
    if fmax is not None and n_bins is None:
        n_bins = (
            jnp.ceil(bins_per_octave * jnp.log2(fmax / fmin)).astype(int).item()
        )  # Calculate the number of bins
        freqs = fmin * 2.0 ** (jnp.r_[0:n_bins] / bins_per_octave)

    elif fmax is None and n_bins is not None:
        freqs = fmin * 2.0 ** (jnp.r_[0:n_bins] / bins_per_octave)

    else:
        # warnings.warn("If fmax is given, n_bins will be ignored", SyntaxWarning)
        assert fmax is not None
        n_bins = (
            jnp.ceil(bins_per_octave * jnp.log2(fmax / fmin)).astype(int).item()
        )  # Calculate the number of bins
        freqs = fmin * 2.0 ** (jnp.r_[0:n_bins] / bins_per_octave)
    if jnp.max(freqs) > sr / 2 and topbin_check:
        raise ValueError(
            "The top bin {}Hz has exceeded the Nyquist frequency, \
                          please reduce the n_bins".format(jnp.max(freqs))
        )

    assert n_bins is not None

    if dtype is not None:
        freqs = freqs.astype(dtype)

    alpha = 2.0 ** (1.0 / bins_per_octave) - 1.0
    lengths = jnp.ceil(Q * sr / (freqs + gamma / alpha)).astype(int)

    # get max window length depending on gamma value
    max_len = lengths.max().item()
    fft_len = 2 ** jnp.ceil(jnp.log2(max_len)).astype(int).item()

    complex_dtype = jnp.result_type(1j, freqs.dtype)

    kernels = jnp.zeros((n_bins, fft_len), dtype=complex_dtype)

    for k in range(n_bins):
        freq = freqs[k]
        curr_len = lengths[k]

        start = jnp.where(
            curr_len % 2 == 1,
            jnp.ceil(fft_len / 2.0 - curr_len / 2.0).astype(int) - 1,
            jnp.ceil(fft_len / 2.0 - curr_len / 2.0).astype(int),
        )

        win = util.get_window(window, int(curr_len), fftbins=True)
        sig = (
            win
            * jnp.exp(
                jnp.arange(-curr_len // 2, curr_len // 2) * 1j * 2 * jnp.pi * freq / sr
            )
            / curr_len
        )

        if norm:  # Normalizing the filter # Trying to normalize like librosa
            sig = sig / jnp.linalg.norm(sig, norm)

        kernels = kernels.at[k, start : start + int(curr_len)].set(sig)

    return kernels, lengths, freqs


def cqt(
    x: Float[Array, " n_samples"],
    /,
    sr: float,
    hop_length=512,
    fmin: float = 32.70,
    fmax: float | None = None,
    n_bins: int = 84,
    bins_per_octave: int = 12,
    filter_scale: float | int = 1.0,
    norm_kernels: float | int = 1,
    power: int | float | None = 2.0,
    window: str | float | tuple = "hann",
    center: bool = True,
    normalization_type: Literal["librosa", "convolutional", "wrap"] = "librosa",
    pad_kwargs=dict(),
) -> Inexact[Array, " n_bins n_frames"]:
    """Compute the Constant-Q Transform (CQT) of a time-domain signal.

    The CQT is a time-frequency representation with logarithmically-spaced frequency bins,
    making it well-suited for music analysis. This is a convenience wrapper that calls
    [vqt][korvax.transforms.vqt] with gamma=0.
    """
    return vqt(
        x,
        sr=sr,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax,
        n_bins=n_bins,
        gamma=0.0,
        bins_per_octave=bins_per_octave,
        filter_scale=filter_scale,
        norm_kernels=norm_kernels,
        power=power,
        window=window,
        center=center,
        normalization_type=normalization_type,
        pad_kwargs=pad_kwargs,
    )


def vqt(
    x: Float[Array, " n_samples"],
    /,
    sr: float,
    hop_length=512,
    fmin: float = 32.70,
    fmax: float | None = None,
    n_bins: int = 84,
    gamma: float = 0.0,
    bins_per_octave: int = 12,
    filter_scale: float | int = 1.0,
    norm_kernels: float | int = 1,
    power: int | float | None = 2.0,
    window: str | float | tuple = "hann",
    center: bool = True,
    normalization_type: Literal["librosa", "convolutional", "wrap"] = "librosa",
    pad_kwargs=dict(),
) -> Float[Array, " n_bins n_frames"]:
    """Compute the Variable-Q Transform (VQT) of a time-domain signal.

    The VQT is a generalization of the Constant-Q Transform (CQT) that allows for variable
    bandwidth via the gamma parameter. When gamma=0, this is equivalent to the CQT.

    Args:
        x: Input signal.
        sr: Sample rate of the input signal.
        hop_length: Hop (step) length between adjacent frames.
        fmin: Minimum frequency (Hz).
        fmax: Maximum frequency (Hz). If None, determined by `n_bins`.
        n_bins: Number of frequency bins. Ignored if `fmax` is provided.
        gamma: Bandwidth offset parameter. When gamma=0, this reduces to CQT.
        bins_per_octave: Number of bins per octave.
        filter_scale: Scale factor for filter bandwidths.
        norm_kernels: Normalization mode for the filter kernels (p-norm to use).
        power: Exponent for the magnitude spectrogram. If 2.0, returns power spectrogram.
            If None, returns complex VQT coefficients.
        window: Window specification (see [get_window][korvax.util.get_window]).
        center: If True, pad the input so that frames are centered on their timestamps.
        normalization_type: Type of normalization to apply ("librosa", "convolutional", or "wrap").
        pad_kwargs: Additional keyword arguments forwarded to [pad_center][korvax.util.pad_center].

    Returns:
        VQT coefficients.
    """
    with jax.ensure_compile_time_eval():
        Q = float(filter_scale) / (2 ** (1 / bins_per_octave) - 1)
        vqt_kernels, lengths, _ = create_vqt_kernels(
            Q=Q,
            sr=sr,
            fmin=fmin,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
            norm=norm_kernels,
            window=window,
            fmax=fmax,
            gamma=gamma,
            dtype=x.dtype,
        )

        n_bins, fft_len = vqt_kernels.shape
        vqt_kernels = jnp.concat(
            [vqt_kernels.real[:, None, :], vqt_kernels.imag[:, None, :]], axis=0
        )

        if normalization_type == "librosa":
            norm_factor = jnp.tile(jnp.sqrt(lengths)[:, None], (2, 1))
        elif normalization_type == "convolutional":
            norm_factor = 1
        elif normalization_type == "wrap":
            norm_factor = 2

    if center:
        x = util.pad_center(x, len(x) + fft_len, **pad_kwargs)

    out = lax.conv_general_dilated(
        lhs=x[None, None, :],
        rhs=vqt_kernels,
        window_strides=(hop_length,),
        padding="VALID",
    ).squeeze(axis=0)

    out = out * norm_factor

    if power is None:
        return out[:n_bins, :] - 1j * out[n_bins:, :]

    elif power == 2:
        return out[:n_bins, :] ** 2 + out[n_bins:, :] ** 2

    return (out[:n_bins, :] ** 2 + out[n_bins:, :] ** 2) ** (power / 2)
