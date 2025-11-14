from typing import overload

import jax.lax as lax
import jax.numpy as jnp


from jaxtyping import Float, Array


@overload
def lfilter(
    x: Float[Array, " n_samples"],
    a: Float[Array, " n_a"],
    b: Float[Array, " n_b"],
    zi: None = None,
    /,
    unroll: int = 8,
) -> Float[Array, " n_samples"]: ...


@overload
def lfilter(
    x: Float[Array, " n_samples"],
    a: Float[Array, " n_a"],
    b: Float[Array, " n_b"],
    zi: Float[Array, " order"],
    /,
    unroll: int = 8,
) -> tuple[Float[Array, " n_samples"], Float[Array, " order"]]: ...


def lfilter(
    x: Float[Array, " n_samples"],
    a: Float[Array, " n_a"],
    b: Float[Array, " n_b"],
    zi: Float[Array, " order"] | None = None,
    /,
    unroll: int = 8,
) -> (
    tuple[Float[Array, " n_samples"], Float[Array, " order"]]
    | Float[Array, " n_samples"]
):
    b = b / a[0]
    a = a / a[0]

    order = max(len(a), len(b)) - 1
    if zi is None:
        return_zi = False
        zi = jnp.zeros((order,), dtype=x.dtype)
    else:
        return_zi = True
        if zi.shape[-1] != order:
            raise ValueError(
                f"Initial conditions zi must have length {order}, but got {zi.shape[-1]}"
            )

    if b.shape[-1] < order + 1:
        b = jnp.pad(b, ((0, 0), (0, order + 1 - b.shape[-1])))
    if a.shape[-1] < order + 1:
        a = jnp.pad(a, ((0, 0), (0, order + 1 - a.shape[-1])))

    def step_fn(carry, xn):
        yn = b[0] * xn + carry[0]

        carry = jnp.roll(carry, shift=-1)
        carry = carry.at[-1].set(0)
        carry = carry + b[1:] * xn - a[1:] * yn
        return carry, yn

    zi, y = lax.scan(step_fn, zi, x, unroll=unroll)

    if return_zi:
        return y, zi
    else:
        return y


@overload
def time_varying_all_pole(
    x: Float[Array, " n_samples"],
    a: Float[Array, "n_samples order"],
    zi: None = None,
    /,
    unroll: int = 8,
) -> Float[Array, " n_samples"]: ...


@overload
def time_varying_all_pole(
    x: Float[Array, " n_samples"],
    a: Float[Array, "n_samples order"],
    zi: Float[Array, " order"],
    /,
    unroll: int = 8,
) -> tuple[Float[Array, " n_samples"], Float[Array, " order"]]: ...


def time_varying_all_pole(
    x: Float[Array, " n_samples"],
    a: Float[Array, "n_samples order"],
    zi: Float[Array, " order"] | None = None,
    /,
    unroll: int = 8,
) -> (
    tuple[Float[Array, " n_samples"], Float[Array, " order"]]
    | Float[Array, " n_samples"]
):
    order = a.shape[-1]
    if zi is None:
        return_zi = False
        zi = jnp.zeros((order,), dtype=x.dtype)
    else:
        return_zi = True
        if zi.shape[-1] != order:
            raise ValueError(
                f"Initial conditions zi must have length {order}, but got {zi.shape[-1]}"
            )

    def step(carry, inputs):
        xn, an = inputs
        yn = xn - jnp.sum(an * carry)
        carry = jnp.roll(carry, shift=1)
        return carry.at[0].set(yn), yn

    zi, y = lax.scan(step, zi, (x, a), unroll=unroll)

    if return_zi:
        return y, zi
    else:
        return y
