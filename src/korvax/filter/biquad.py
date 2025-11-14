from typing import overload

import jax.lax as lax
import jax.numpy as jnp

from jaxtyping import Float, Array


@overload
def biquad(
    x: Float[Array, " n_samples"],
    b0: Float[Array, ""],
    b1: Float[Array, ""],
    b2: Float[Array, ""],
    a1: Float[Array, ""],
    a2: Float[Array, ""],
    zi: None = None,
) -> Float[Array, " n_samples"]: ...


@overload
def biquad(
    x: Float[Array, " n_samples"],
    b0: Float[Array, ""],
    b1: Float[Array, ""],
    b2: Float[Array, ""],
    a1: Float[Array, ""],
    a2: Float[Array, ""],
    zi: Float[Array, " 2"],
) -> tuple[Float[Array, " n_samples"], Float[Array, " 2"]]: ...


def biquad(
    x: Float[Array, " n_samples"],
    b0: Float[Array, ""],
    b1: Float[Array, ""],
    b2: Float[Array, ""],
    a1: Float[Array, ""],
    a2: Float[Array, ""],
    zi: Float[Array, " 2"] | None = None,
) -> tuple[Float[Array, " n_samples"], Float[Array, " 2"]] | Float[Array, " n_samples"]:
    if zi is None:
        return_zi = False
        zi = jnp.zeros(x.shape[:-1] + (2,), dtype=x.dtype)
    else:
        return_zi = True
        if zi.shape[-1] != 2:
            raise ValueError(
                f"Initial conditions zi must have length 2, but got {zi.shape[-1]}"
            )

    def step_fn(carry, xn):
        yn = b0 * xn + carry[0]

        c1 = carry[1] + b1 * xn - a1 * yn
        c2 = b2 * xn - a2 * yn
        return jnp.r_[c1, c2], yn

    zi, y = lax.scan(step_fn, zi, x)

    if return_zi:
        assert zi is not None
        return y, zi
    else:
        return y


@overload
def time_varying_biquad(
    x: Float[Array, " n_samples"],
    b0: Float[Array, " n_samples"],
    b1: Float[Array, " n_samples"],
    b2: Float[Array, " n_samples"],
    a1: Float[Array, " n_samples"],
    a2: Float[Array, " n_samples"],
    zi: None = None,
) -> Float[Array, " n_samples"]: ...


@overload
def time_varying_biquad(
    x: Float[Array, " n_samples"],
    b0: Float[Array, " n_samples"],
    b1: Float[Array, " n_samples"],
    b2: Float[Array, " n_samples"],
    a1: Float[Array, " n_samples"],
    a2: Float[Array, " n_samples"],
    zi: Float[Array, " 4"],
) -> tuple[Float[Array, " n_samples"], Float[Array, " 4"]]: ...


def time_varying_biquad(
    x: Float[Array, " n_samples"],
    b0: Float[Array, " n_samples"],
    b1: Float[Array, " n_samples"],
    b2: Float[Array, " n_samples"],
    a1: Float[Array, " n_samples"],
    a2: Float[Array, " n_samples"],
    zi: Float[Array, " 4"] | None = None,
) -> tuple[Float[Array, " n_samples"], Float[Array, " 4"]] | Float[Array, " n_samples"]:
    if zi is None:
        return_zi = False
        zi = jnp.zeros(x.shape[:-1] + (4,), dtype=x.dtype)
    else:
        return_zi = True
        if zi.shape[-1] != 4:
            raise ValueError(
                f"Initial conditions zi must have length 4, but got {zi.shape[-1]}"
            )

    def step_fn(carry, inputs):
        xn, b0n, b1n, b2n, a1n, a2n = inputs

        yn = (
            b0n * xn + b1n * carry[0] + b2n * carry[1] - a1n * carry[2] - a2n * carry[3]
        )

        return jnp.stack([xn, carry[0], yn, carry[2]], axis=-1), yn

    zi, y = lax.scan(step_fn, zi, (x, b0, b1, b2, a1, a2))

    if return_zi:
        assert zi is not None
        return y, zi
    else:
        return y


@overload
def sosfilt(
    x: Float[Array, " n_samples"],
    b0: Float[Array, " n_sections"],
    b1: Float[Array, " n_sections"],
    b2: Float[Array, " n_sections"],
    a1: Float[Array, " n_sections"],
    a2: Float[Array, " n_sections"],
    zi: None = None,
) -> Float[Array, " n_samples"]: ...


@overload
def sosfilt(
    x: Float[Array, " n_samples"],
    b0: Float[Array, " n_sections"],
    b1: Float[Array, " n_sections"],
    b2: Float[Array, " n_sections"],
    a1: Float[Array, " n_sections"],
    a2: Float[Array, " n_sections"],
    zi: Float[Array, " n_sections 2"],
) -> tuple[Float[Array, " n_samples"], Float[Array, " n_sections 2"]]: ...


def sosfilt(
    x: Float[Array, " n_samples"],
    b0: Float[Array, " n_sections"],
    b1: Float[Array, " n_sections"],
    b2: Float[Array, " n_sections"],
    a1: Float[Array, " n_sections"],
    a2: Float[Array, " n_sections"],
    zi: Float[Array, " n_sections 2"] | None = None,
) -> (
    tuple[Float[Array, " n_samples"], Float[Array, " n_sections 2"]]
    | Float[Array, " n_samples"]
):
    n_sections = b0.shape[-1]

    if zi is None:
        return_zi = False
        zi = jnp.zeros(x.shape[:-1] + (n_sections, 2), dtype=x.dtype)
    else:
        return_zi = True
        if zi.shape[-1] != 2 or zi.shape[-2] != n_sections:
            raise ValueError(
                f"Initial conditions zi must have shape (..., {n_sections}, 2), but got {zi.shape}"
            )

    def step_fn(carry, inputs):
        b0n, b1n, b2n, a1n, a2n, zi = inputs
        return biquad(carry, b0=b0n, b1=b1n, b2=b2n, a1=a1n, a2=a2n, zi=zi)

    y, zi = lax.scan(step_fn, x, (b0, b1, b2, a1, a2, zi))

    if return_zi:
        assert zi is not None
        return y, zi
    else:
        return y


@overload
def time_varying_sosfilt(
    x: Float[Array, " n_samples"],
    b0: Float[Array, " n_sections n_samples"],
    b1: Float[Array, " n_sections n_samples"],
    b2: Float[Array, " n_sections n_samples"],
    a1: Float[Array, " n_sections n_samples"],
    a2: Float[Array, " n_sections n_samples"],
    zi: None = None,
) -> Float[Array, " n_samples"]: ...


@overload
def time_varying_sosfilt(
    x: Float[Array, " n_samples"],
    b0: Float[Array, " n_sections n_samples"],
    b1: Float[Array, " n_sections n_samples"],
    b2: Float[Array, " n_sections n_samples"],
    a1: Float[Array, " n_sections n_samples"],
    a2: Float[Array, " n_sections n_samples"],
    zi: Float[Array, " n_sections 4"],
) -> tuple[Float[Array, " n_samples"], Float[Array, " n_sections 4"]]: ...


def time_varying_sosfilt(
    x: Float[Array, " n_samples"],
    b0: Float[Array, " n_sections n_samples"],
    b1: Float[Array, " n_sections n_samples"],
    b2: Float[Array, " n_sections n_samples"],
    a1: Float[Array, " n_sections n_samples"],
    a2: Float[Array, " n_sections n_samples"],
    zi: Float[Array, " n_sections 4"] | None = None,
) -> (
    tuple[Float[Array, " n_samples"], Float[Array, " n_sections 4"]]
    | Float[Array, " n_samples"]
):
    n_sections = b0.shape[-2]

    if zi is None:
        return_zi = False
        zi = jnp.zeros(x.shape[:-1] + (n_sections, 4), dtype=x.dtype)
    else:
        return_zi = True
        if zi.shape[-1] != 4 or zi.shape[-2] != n_sections:
            raise ValueError(
                f"Initial conditions zi must have shape (..., {n_sections}, 4), but got {zi.shape}"
            )

    def step_fn(carry, inputs):
        b0n, b1n, b2n, a1n, a2n, zi = inputs
        return time_varying_biquad(carry, b0=b0n, b1=b1n, b2=b2n, a1=a1n, a2=a2n, zi=zi)

    y, zi = lax.scan(step_fn, x, (b0, b1, b2, a1, a2, zi))

    if return_zi:
        assert zi is not None
        return y, zi
    else:
        return y
