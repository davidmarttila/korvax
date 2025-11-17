from collections.abc import Callable
from jaxtyping import Float, ArrayLike, Array, Inexact
import jax.numpy as jnp


def midi_to_hz(notes: Float[ArrayLike, "*dims"]) -> Float[Array, "*dims"]:
    """Convert MIDI note numbers to frequencies in Hz.

    Args:
        notes: MIDI note numbers (A440 = 69).

    Returns:
        Frequencies in Hz.
    """
    return 440.0 * (2.0 ** ((jnp.asarray(notes) - 69.0) / 12.0))


def hz_to_midi(frequencies: Float[ArrayLike, "*dims"]) -> Float[Array, "*dims"]:
    """Convert frequencies in Hz to MIDI note numbers.

    Args:
        frequencies: Frequencies in Hz.

    Returns:
        MIDI note numbers (A440 = 69).
    """
    return 12 * (jnp.log2(frequencies) - jnp.log2(440.0)) + 69.0


def cents_to_hz(
    cents: Float[ArrayLike, "*dims"], /, hz_ref: float
) -> Float[Array, "*dims"]:
    """Convert cents to frequencies in Hz relative to a reference frequency.

    Args:
        cents: Cents relative to reference frequency.
        hz_ref: Reference frequency in Hz.

    Returns:
        Frequencies in Hz.
    """
    return hz_ref * 2 ** (jnp.asarray(cents) / 1200.0)


def hz_to_cents(
    frequencies: Float[ArrayLike, "*dims"], /, hz_ref: float
) -> Float[Array, "*dims"]:
    """Convert frequencies in Hz to cents relative to a reference frequency.

    Args:
        frequencies: Frequencies in Hz.
        hz_ref: Reference frequency in Hz.

    Returns:
        Cents relative to reference frequency.
    """
    return 1200.0 * (jnp.log2(frequencies) - jnp.log2(hz_ref))


def mel_to_hz(
    mels: Float[ArrayLike, "*dims"], /, htk: bool = False
) -> Float[Array, "*dims"]:
    """Convert mel scale to frequencies in Hz.

    Args:
        mels: Mel-scale values.
        htk: If True, use HTK formula. Otherwise, use Slaney formula.

    Returns:
        Frequencies in Hz.
    """
    mels = jnp.asarray(mels)

    if htk:
        return 700.0 * (10 ** (mels / 2595.0) - 1.0)

    # Fill in the linear part
    f_min = 0.0
    f_sp = 200.0 / 3

    frequencies = f_min + f_sp * mels

    # Fill in the log-scale part
    min_log_hz = 1000.0  # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
    logstep = jnp.log(6.4) / 27.0  # step size for log region

    return jnp.where(
        mels >= min_log_mel,
        min_log_hz * jnp.exp(logstep * (mels - min_log_mel)),
        frequencies,
    )


def hz_to_mel(
    frequencies: Float[ArrayLike, "*dims"], /, htk: bool = False
) -> Float[Array, "*dims"]:
    """Convert frequencies in Hz to mel scale.

    Args:
        frequencies: Frequencies in Hz.
        htk: If True, use HTK formula. Otherwise, use Slaney formula.

    Returns:
        Mel-scale values.
    """
    frequencies = jnp.asarray(frequencies)

    if htk:
        return 2595.0 * jnp.log10(1.0 + frequencies / 700.0)

    # Fill in the linear part
    f_min = 0.0
    f_sp = 200.0 / 3

    mels = (frequencies - f_min) / f_sp

    # Fill in the log-scale part
    min_log_hz = 1000.0  # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
    logstep = jnp.log(6.4) / 27.0  # step size for log region

    return jnp.where(
        frequencies >= min_log_hz,
        min_log_mel + jnp.log(frequencies / min_log_hz) / logstep,
        mels,
    )


def fft_frequencies(
    *, sr: float = 22050, n_fft: int = 2048
) -> Float[Array, " {n_fft}//2+1"]:
    """Compute the center frequencies of FFT bins.

    Args:
        sr: Sample rate of the audio signal.
        n_fft: FFT size (number of samples per frame).

    Returns:
        Center frequencies of FFT bins in Hz.
    """
    return jnp.fft.rfftfreq(n=n_fft, d=1.0 / sr)


def cqt_frequencies(
    n_bins: int, /, fmin: float, bins_per_octave: int = 12, tuning: float = 0.0
) -> Float[Array, " {n_bins}"]:
    """Compute the center frequencies of constant-Q transform bins.

    Args:
        n_bins: Number of frequency bins.
        fmin: Minimum frequency (Hz).
        bins_per_octave: Number of bins per octave.
        tuning: Tuning offset in fractions of a bin.

    Returns:
        Center frequencies of CQT bins in Hz.
    """
    correction = 2.0 ** (tuning / bins_per_octave)
    frequencies = 2.0 ** (jnp.arange(0, n_bins) / bins_per_octave)

    return correction * fmin * frequencies


def mel_frequencies(
    n_mels: int = 128, /, fmin: float = 0.0, fmax: float = 11025.0, htk: bool = False
) -> Float[Array, " {n_mels}"]:
    """Compute an array of mel-spaced frequencies.

    Args:
        n_mels: Number of mel bands.
        fmin: Minimum frequency (Hz).
        fmax: Maximum frequency (Hz).
        htk: If True, use HTK formula. Otherwise, use Slaney formula.

    Returns:
        Array of frequencies in Hz.
    """
    mels = jnp.linspace(hz_to_mel(fmin, htk=htk), hz_to_mel(fmax, htk=htk), n_mels)
    return mel_to_hz(mels, htk=htk)


def A_weighting(
    frequencies: Float[ArrayLike, " n_freqs"], /, min_db: float | None = -80.0
) -> Float[Array, " n_freqs"]:
    """Compute A-weighting curve for given frequencies.

    Args:
        frequencies: Frequencies in Hz.
        min_db: Minimum decibel value for clipping. If None, no clipping applied.

    Returns:
        A-weighting values in dB.
    """
    f = jnp.asarray(frequencies) ** 2
    const = jnp.array([12194.217, 20.598997, 107.65265, 737.86223]) ** 2.0
    weights: jnp.ndarray = 2.0 + 20.0 * (
        jnp.log10(const[0])
        + 2 * jnp.log10(f)
        - jnp.log10(f + const[0])
        - jnp.log10(f + const[1])
        - 0.5 * jnp.log10(f + const[2])
        - 0.5 * jnp.log10(f + const[3])
    )

    if min_db is None:
        return weights
    else:
        return jnp.maximum(min_db, weights)


def B_weighting(
    frequencies: Float[ArrayLike, " n_freqs"], /, min_db: float | None = -80.0
) -> Float[Array, " n_freqs"]:
    """Compute B-weighting curve for given frequencies.

    Args:
        frequencies: Frequencies in Hz.
        min_db: Minimum decibel value for clipping. If None, no clipping applied.

    Returns:
        B-weighting values in dB.
    """
    f = jnp.asarray(frequencies) ** 2
    const = jnp.array([12194.217, 20.598997, 158.48932]) ** 2.0
    weights: jnp.ndarray = 0.17 + 20.0 * (
        jnp.log10(const[0])
        + 1.5 * jnp.log10(f)
        - jnp.log10(f + const[0])
        - jnp.log10(f + const[1])
        - 0.5 * jnp.log10(f + const[2])
    )

    if min_db is None:
        return weights
    else:
        return jnp.maximum(min_db, weights)


def C_weighting(
    frequencies: Float[ArrayLike, " n_freqs"], /, min_db: float | None = -80.0
) -> Float[Array, " n_freqs"]:
    """Compute C-weighting curve for given frequencies.

    Args:
        frequencies: Frequencies in Hz.
        min_db: Minimum decibel value for clipping. If None, no clipping applied.

    Returns:
        C-weighting values in dB.
    """
    f = jnp.asarray(frequencies) ** 2.0
    const = jnp.array([12194.217, 20.598997]) ** 2.0
    weights: jnp.ndarray = 0.062 + 20.0 * (
        jnp.log10(const[0])
        + jnp.log10(f)
        - jnp.log10(f + const[0])
        - jnp.log10(f + const[1])
    )

    if min_db is None:
        return weights
    else:
        return jnp.maximum(min_db, weights)


def D_weighting(
    frequencies: Float[ArrayLike, " n_freqs"], /, min_db: float | None = -80.0
) -> Float[Array, " n_freqs"]:
    """Compute D-weighting curve for given frequencies.

    Args:
        frequencies: Frequencies in Hz.
        min_db: Minimum decibel value for clipping. If None, no clipping applied.

    Returns:
        D-weighting values in dB.
    """
    f = jnp.asarray(frequencies) ** 2
    const = jnp.array([8.3046305e-3, 1018.7, 1039.6, 3136.5, 3424, 282.7, 1160]) ** 2.0
    weights = 20.0 * (
        0.5 * jnp.log10(f)
        - jnp.log10(const[0])
        + 0.5
        * (
            +jnp.log10((const[1] - f) ** 2 + const[2] * f)
            - jnp.log10((const[3] - f) ** 2 + const[4] * f)
            - jnp.log10(const[5] + f)
            - jnp.log10(const[6] + f)
        )
    )

    if min_db is None:
        return weights
    else:
        return jnp.maximum(min_db, weights)


def power_to_db(
    S: Inexact[ArrayLike, "*dims"],
    /,
    ref: Float[ArrayLike, ""]
    | Callable[[Float[ArrayLike, "*"]], Float[ArrayLike, ""]] = 1.0,
    amin: float = 1e-10,
    top_db: float | None = 80.0,
) -> Float[Array, "*dims"]:
    """Convert a power spectrogram to decibel scale.

    Args:
        S: Input power spectrogram.
        ref: Reference value for decibel calculation. Can be a scalar or callable
            that computes a reference from the input.
        amin: Minimum threshold for input values.
        top_db: Maximum decibel range. Values below `max - top_db` are clipped.
            If None, no clipping applied.

    Returns:
        Power spectrogram in dB.
    """
    if jnp.issubdtype(jnp.result_type(S), jnp.complexfloating):
        power = jnp.abs(S)
    else:
        power = jnp.asarray(S)

    if callable(ref):
        ref_value = ref(power)
    else:
        ref_value = jnp.abs(ref)

    log_spec = 10.0 * jnp.log10(jnp.maximum(amin, power))
    log_spec -= 10.0 * jnp.log10(jnp.maximum(amin, ref_value))

    if top_db is not None:
        log_spec = jnp.maximum(log_spec, log_spec.max() - top_db)

    return log_spec


def db_to_power(
    S_db: Float[ArrayLike, "*dims"],
    /,
    ref: Float[ArrayLike, ""] = 1.0,
) -> Float[Array, "*dims"]:
    """Convert a decibel-scale spectrogram to power scale.

    Args:
        S_db: Input spectrogram in dB.
        ref: Reference value for decibel calculation.

    Returns:
        Power spectrogram.
    """
    return jnp.asarray(ref) * (10.0 ** (jnp.asarray(S_db) / 10.0))


def amplitude_to_db(
    S: Inexact[ArrayLike, "*dims"],
    /,
    ref: Float[ArrayLike, ""]
    | Callable[[Float[ArrayLike, "*"]], Float[ArrayLike, ""]] = 1.0,
    amin: float = 1e-8,
    top_db: float | None = 80.0,
) -> Float[Array, "*dims"]:
    """Convert an amplitude spectrogram to decibel scale.

    Args:
        S: Input amplitude spectrogram.
        ref: Reference value for decibel calculation. Can be a scalar or callable
            that computes a reference from the input.
        amin: Minimum threshold for input values.
        top_db: Maximum decibel range. Values below `max - top_db` are clipped.
            If None, no clipping applied.

    Returns:
        Amplitude spectrogram in dB.
    """
    if jnp.issubdtype(jnp.result_type(S), jnp.complexfloating):
        mag = jnp.abs(S)
    else:
        mag = jnp.asarray(S)
    if callable(ref):
        ref_value = ref(mag)
    else:
        ref_value = jnp.abs(ref)

    power = mag**2

    return power_to_db(power, ref=ref_value**2, amin=amin**2, top_db=top_db)  # pyright: ignore[reportArgumentType]


def db_to_amplitude(
    S_db: Float[ArrayLike, "*dims"],
    /,
    ref: float = 1.0,
) -> Float[Array, "*dims"]:
    """Convert a decibel-scale spectrogram to amplitude scale.

    Args:
        S_db: Input spectrogram in dB.
        ref: Reference value for decibel calculation.

    Returns:
        Amplitude spectrogram.
    """
    return ref * (10.0 ** (jnp.asarray(S_db) / 20.0))
