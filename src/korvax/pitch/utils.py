import jax
from jaxtyping import Float, Array
import jax.numpy as jnp

def midi_to_hz(notes: Float[Array, "*axes"], a4_hz: float=440.0, a4_midi: float=69.0) -> Float[Array, "*axes"]:
    return a4_hz * (2.0 ** (notes - a4_midi) / 12.0)

def hz_to_midi(frequencies: Float[Array, "*axes"], a4_hz: float=440.0, a4_midi: float=69.0) -> Float[Array, "*axes"]:
    return 12 * (jnp.log2(frequencies) - jnp.log2(a4_hz)) + a4_midi

def cents_to_hz(cents: Float[Array, "*axes"], hz_ref: float) -> Float[Array, "*axes"]:
    return hz_ref * 2 ** (cents / 1200.0)

def hz_to_cents(frequencies: Float[Array, "*axes"], hz_ref: float) -> Float[Array, "*axes"]:
    return 1200.0 * (jnp.log2(frequencies) - jnp.log2(hz_ref))

