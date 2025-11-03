import jax
from jax import lax, numpy as jnp
from jaxtyping import Float, Array
from typing import Any

import jax._src.scipy.signal


def frame(
    x: Float[Array, "*channels n_samples"],
    /,
    frame_length: int,
    hop_length: int,
) -> Float[
    Array,
    "*channels {frame_length} n_frames=1+(n_samples-{frame_length})//{hop_length}",
]:
    """Slice a JAX array into overlapping frames.

    Args:
        x: Input array.
        frame_length: Length of each frame.
        hop_length: Number of samples between adjacent frame starts.

    Returns:
        Array with the last axis sliced into overlapping frames.
    """
    n_samples = x.shape[-1]
    n_frames = 1 + (n_samples - frame_length) // hop_length

    return jax.vmap(
        lax.dynamic_slice_in_dim, in_axes=(None, 0, None, None), out_axes=-1
    )(x, jnp.arange(n_frames) * hop_length, frame_length, -1)


def overlap_and_add(
    x: Float[Array, "*channels frame_length n_frames"],
    hop_length: int,
) -> Float[Array, "*channels n_samples"]:
    """Construct a signal from overlapping frames with overlap-and-add.

    Args:
        x: Input array containing overlapping frames.
        hop_length: Number of samples between adjacent frame starts.

    Returns:
        Constructed time-domain signal.
    """
    return jax._src.scipy.signal._overlap_and_add(x.swapaxes(-2, -1), hop_length)


def pad_center(
    x: Float[Array, "*channels n_samples"], /, size: int, **pad_kwargs: Any
) -> Float[Array, "*channels {size}"]:
    """Pad the input array on both sides to center it in a new array of given size.

    Args:
        x: Input array.
        size: Desired size of the last axis after padding.
        **pad_kwargs: Additional keyword arguments forwarded to [`jax.numpy.pad`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.pad.html).

    Returns:
        Array with the last axis center-padded to the desired size.
    """
    n_samples = x.shape[-1]

    lpad = int((size - n_samples) // 2)

    lengths = [(0, 0)] * x.ndim
    lengths[-1] = (lpad, int(size - n_samples - lpad))

    return jnp.pad(x, lengths, **pad_kwargs)


def fix_length(
    x: Float[Array, "*channels n_samples"], /, size: int, **pad_kwargs: Any
) -> Float[Array, "*channels {size}"]:
    """Fix the length of the input array to a given size by either trimming or padding.

    Args:
        x: Input array.
        size: Desired size of the last axis after fixing length.
        **pad_kwargs: Additional keyword arguments forwarded to [`jax.numpy.pad`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.pad.html).

    Returns:
        Array with the last axis fixed to the desired size.
    """
    n_samples = x.shape[-1]

    if n_samples < size:
        lengths = [(0, 0)] * x.ndim
        lengths[-1] = (0, size - n_samples)
        return jnp.pad(x, lengths, **pad_kwargs)
    else:
        return x[..., :size]
