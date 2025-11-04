from collections.abc import Callable
from jaxtyping import Float, ArrayLike, Array, Inexact
import jax.numpy as jnp


def midi_to_hz(notes: Float[ArrayLike, "*dims"]) -> Float[Array, "*dims"]:
    return 440.0 * (2.0 ** ((jnp.asarray(notes) - 69.0) / 12.0))


def hz_to_midi(frequencies: Float[ArrayLike, "*dims"]) -> Float[Array, "*dims"]:
    return 12 * (jnp.log2(frequencies) - jnp.log2(440.0)) + 69.0


def cents_to_hz(
    cents: Float[ArrayLike, "*dims"], /, hz_ref: float
) -> Float[Array, "*dims"]:
    return hz_ref * 2 ** (jnp.asarray(cents) / 1200.0)


def hz_to_cents(
    frequencies: Float[ArrayLike, "*dims"], /, hz_ref: float
) -> Float[Array, "*dims"]:
    return 1200.0 * (jnp.log2(frequencies) - jnp.log2(hz_ref))


def mel_to_hz(
    mels: Float[ArrayLike, "*dims"], /, htk: bool = False
) -> Float[Array, "*dims"]:
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
    return jnp.fft.rfftfreq(n=n_fft, d=1.0 / sr)


def cqt_frequencies(
    n_bins: int, /, fmin: float, bins_per_octave: int = 12, tuning: float = 0.0
) -> Float[Array, " {n_bins}"]:
    correction = 2.0 ** (tuning / bins_per_octave)
    frequencies = 2.0 ** (jnp.arange(0, n_bins) / bins_per_octave)

    return correction * fmin * frequencies


def mel_frequencies(
    n_mels: int = 128, /, fmin: float = 0.0, fmax: float = 11025.0, htk: bool = False
) -> Float[Array, " {n_mels}"]:
    mels = jnp.linspace(hz_to_mel(fmin, htk=htk), hz_to_mel(fmax, htk=htk), n_mels)
    return mel_to_hz(mels, htk=htk)


def A_weighting(
    frequencies: Float[ArrayLike, " n_freqs"], /, min_db: float | None = -80.0
) -> Float[Array, " n_freqs"]:
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
    return jnp.asarray(ref) * (10.0 ** (jnp.asarray(S_db) / 10.0))


def amplitude_to_db(
    S: Inexact[ArrayLike, "*dims"],
    /,
    ref: Float[ArrayLike, ""]
    | Callable[[Float[ArrayLike, "*"]], Float[ArrayLike, ""]] = 1.0,
    amin: float = 1e-8,
    top_db: float | None = 80.0,
) -> Float[Array, "*dims"]:
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
    return ref * (10.0 ** (jnp.asarray(S_db) / 20.0))
