from typing import Any
import jax.numpy as jnp
from jaxtyping import Float, Array, ArrayLike, Complex, ScalarLike
import equinox as eqx

from korvax.util import pad_center, frame, get_window


def stft(
    x: Float[Array, "*channels n_samples"],
    /,
    n_fft: int,
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
            Ignored if `window` is a JAX array.
        window: Either a 1d JAX array containing the window to apply to each frame,
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
        x = pad_center(x, size=x.shape[-1] + n_fft, **pad_kwargs)

    frames = frame(x, frame_length=n_fft, hop_length=hop_length)

    if eqx.is_array(window):
        fft_window = jnp.asarray(window)
    else:
        fft_window = get_window(window, win_length, fftbins=True)  # pyright: ignore[reportArgumentType]

    win_dims = [1] * frames.ndim
    win_dims[-2] = len(fft_window)
    fft_window = fft_window.reshape(*win_dims)

    return jnp.fft.rfft(frames * fft_window, n=n_fft, axis=-2)
