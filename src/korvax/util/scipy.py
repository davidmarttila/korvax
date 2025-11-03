import scipy
import jax.numpy as jnp
from typing import Any

from jaxtyping import Float, Array


def get_window(
    window: str | float | tuple, Nx: int, fftbins: bool = True
) -> Float[Array, " {Nx}"]:
    """Return the output of [`scipy.signal.get_window`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html) as a JAX array.

    Args:
        window (str | float | tuple): Window specification.
        Nx (int): Length of the returned window.
        fftbins (bool, optional): If `True`, return a periodic window for FFT analysis.
            If `False`, return a symmetric window for filter design. Default: `True`.

    Returns:
        The window as a JAX array.
    """
    win = scipy.signal.get_window(window, Nx, fftbins=fftbins)
    return jnp.asarray(win)
