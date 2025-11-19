import jax.numpy as jnp
from jaxtyping import DTypeLike

from .. import util
from .._typing import _WindowSpec


def create_cqt_kernels(
    Q: float,
    sr: float,
    fmin: float,
    n_bins: int | None = 84,
    bins_per_octave: int = 12,
    norm: float | int = 1,
    window: _WindowSpec = "hann",
    fmax: float | None = None,
    topbin_check: bool = True,
    gamma: float = 0.0,
    dtype: DTypeLike | None = None,
):
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
