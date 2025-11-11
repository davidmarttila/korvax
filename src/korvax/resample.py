"""
JAX port of https://docs.pytorch.org/audio/main/generated/torchaudio.functional.resample.html
"""

import math
import warnings
import jax
import jax.numpy as jnp
from jax.scipy.special import i0
from jaxtyping import DTypeLike, Float, Array


def _get_sinc_resample_kernel(
    orig_sr: int,
    target_sr: int,
    gcd: int,
    lowpass_filter_width: int = 6,
    rolloff: float = 0.99,
    resampling_method: str = "sinc_interp_hann",
    beta: float | None = None,
    dtype: DTypeLike = jnp.float32,
) -> tuple[Float[Array, "out_ch in_ch width"], int]:
    """Calculates a sinc resampling kernel."""
    if not (isinstance(orig_sr, int) and isinstance(target_sr, int)):
        raise ValueError(
            "Frequencies must be integers. To work around this, manually convert "
            "frequencies to integers that maintain their ratio. "
            "E.g., for 44100 -> 5512.5, use orig_sr=8, target_sr=1."
        )

    if resampling_method in ["sinc_interpolation", "kaiser_window"]:
        method_map = {
            "sinc_interpolation": "sinc_interp_hann",
            "kaiser_window": "sinc_interp_kaiser",
        }
        warnings.warn(
            f'"{resampling_method}" is deprecated and will be removed. '
            f'Please use "{method_map[resampling_method]}" instead.'
        )
        resampling_method = method_map[resampling_method]
    elif resampling_method not in ["sinc_interp_hann", "sinc_interp_kaiser"]:
        raise ValueError(f"Invalid resampling method: {resampling_method}")

    orig_sr = orig_sr // gcd
    target_sr = target_sr // gcd

    if lowpass_filter_width <= 0:
        raise ValueError("Low pass filter width should be positive.")

    base_freq = min(orig_sr, target_sr) * rolloff
    width = math.ceil(lowpass_filter_width * orig_sr / base_freq)

    # Calculate indices for the kernel
    idx = jnp.arange(-width, width + orig_sr, dtype=float) / orig_sr

    # Create time steps for the kernel
    t = (jnp.arange(0, -target_sr, -1, dtype=float)[:, None] / target_sr) + idx[None, :]
    t *= base_freq
    t = jnp.clip(t, min=-lowpass_filter_width, max=lowpass_filter_width)

    # Apply window function
    if resampling_method == "sinc_interp_hann":
        window = jnp.cos(t * math.pi / lowpass_filter_width / 2) ** 2
    else:  # sinc_interp_kaiser
        if beta is None:
            beta = 14.769656459379492
        beta_tensor = jnp.array(float(beta))
        window = i0(beta_tensor * jnp.sqrt(1 - (t / lowpass_filter_width) ** 2)) / i0(
            beta_tensor
        )

    # Calculate sinc kernel
    t *= math.pi
    scale = base_freq / orig_sr
    kernels = jnp.where(t == 0, 1.0, jnp.sin(t) / t)
    kernels *= window * scale

    # Reshape for convolution and cast to final dtype
    # Shape: (out_channels, in_channels, width)
    kernels = jnp.expand_dims(kernels, axis=1).astype(dtype)
    return kernels, width


def _apply_sinc_resample_kernel(
    waveform: Float[Array, "*batch n_samples"],
    orig_sr: int,
    target_sr: int,
    gcd: int,
    kernel: Float[Array, "out_ch in_ch width"],
    width: int,
) -> Float[Array, "*batch new_n_samples"]:
    """Applies the sinc resampling kernel to a waveform."""
    if not jnp.issubdtype(waveform.dtype, jnp.floating):
        raise TypeError(
            f"Expected floating point type for waveform, but received {waveform.dtype}."
        )

    orig_sr = orig_sr // gcd
    target_sr = target_sr // gcd

    # Reshape waveform to (batch, time)
    shape = waveform.shape
    waveform_2d = waveform.reshape(-1, shape[-1])
    num_wavs, length = waveform_2d.shape

    # Pad waveform for convolution
    waveform_padded = jnp.pad(waveform_2d, ((0, 0), (width, width + orig_sr)))

    # Add a channel dimension for convolution: (batch, channels, time)
    lhs = jnp.expand_dims(waveform_padded, axis=1)

    # Perform convolution
    resampled = jax.lax.conv_general_dilated(
        lhs=lhs,
        rhs=kernel,
        window_strides=(orig_sr,),
        padding="VALID",
        dimension_numbers=("NCW", "OIW", "NCW"),  # N: batch, C: channels, W: width
    )

    # Reshape and truncate to target length
    # Output of conv is (batch, out_channels, out_time)
    # Transpose to (batch, out_time, out_channels) and reshape to interleave samples
    resampled = resampled.transpose(0, 2, 1).reshape(num_wavs, -1)

    with jax.ensure_compile_time_eval():
        target_length = int(jnp.ceil(target_sr * length / orig_sr))
    resampled = resampled[:, :target_length]

    # Reshape back to original batch dimensions
    return resampled.reshape((*shape[:-1], resampled.shape[-1]))


def resample(
    x: Float[Array, "*channels n_samples"],
    /,
    orig_sr: int,
    target_sr: int,
    lowpass_filter_width: int = 6,
    rolloff: float = 0.99,
    resampling_method: str = "sinc_interp_hann",
    beta: float | None = None,
    scale: bool = False,
) -> Float[Array, "*channels new_n_samples"]:
    """Resample a waveform using sinc interpolation.

    Direct JAX port of [`torchaudio.resample`](https://docs.pytorch.org/audio/main/generated/torchaudio.functional.resample.html)

    Requires sampling rates to be specified as integers.

    Args:
        x: The input signal of dimension `(..., time)`.
        orig_sr: The original frequency of the signal.
        target_sr: The desired frequency.
        lowpass_filter_width: Controls the sharpness of the filter.
            A larger value gives a sharper filter but is less efficient. Defaults to ``6``.
        rolloff: The roll-off frequency of the filter as a
            fraction of the Nyquist frequency. Lower values reduce aliasing but
            also attenuate high frequencies. Defaults to ``0.99``.
        resampling_method: The windowing function to use.
            Options: [``"sinc_interp_hann"``, ``"sinc_interp_kaiser"``].
            Defaults to ``"sinc_interp_hann"``.
        beta: The shape parameter for the Kaiser window.
            Only used if `resampling_method` is ``"sinc_interp_kaiser"``.
            Defaults to ``14.7696...``.

    Returns:
        jax.numpy.ndarray: The waveform at the new frequency, with shape `(..., new_time)`.
    """
    if orig_sr <= 0 or target_sr <= 0:
        raise ValueError("Original and new frequencies must be positive.")

    if orig_sr == target_sr:
        return x

    with jax.ensure_compile_time_eval():
        gcd = math.gcd(orig_sr, target_sr)
        kernel, width = _get_sinc_resample_kernel(
            orig_sr,
            target_sr,
            gcd,
            lowpass_filter_width,
            rolloff,
            resampling_method,
            beta,
            dtype=x.dtype,
        )

    x = _apply_sinc_resample_kernel(x, orig_sr, target_sr, gcd, kernel, width)

    if scale:
        x = x / jnp.sqrt((target_sr / orig_sr))

    return x
