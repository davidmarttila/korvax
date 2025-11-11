from typing import overload

import jax.lax as lax
import jax.numpy as jnp

from jaxtyping import Float, Array


@overload
def lfilter(
    b: Float[Array, " n_b"],
    a: Float[Array, " n_a"],
    x: Float[Array, "*channels n_samples"],
    zi: None = None,
) -> Float[Array, "*channels n_samples"]: ...


@overload
def lfilter(
    b: Float[Array, " n_b"],
    a: Float[Array, " n_a"],
    x: Float[Array, "*channels n_samples"],
    zi: Float[Array, "*channels order"],
) -> tuple[Float[Array, "*channels n_samples"], Float[Array, "*channels order"]]: ...


def lfilter(
    b: Float[Array, " n_b"],
    a: Float[Array, " n_a"],
    x: Float[Array, "*channels n_samples"],
    zi: Float[Array, "*channels order"] | None = None,
) -> (
    tuple[Float[Array, "*channels n_samples"], Float[Array, "*channels order"]]
    | Float[Array, "*channels n_samples"]
):
    b = b / a[0]
    a = a / a[0]

    order = max(len(a), len(b)) - 1
    if zi is None:
        return_zi = False
        zi = jnp.zeros(x.shape[:-1] + (order,), dtype=x.dtype)
    else:
        return_zi = True
        if zi.shape[-1] != order:
            raise ValueError(
                f"Initial conditions zi must have length {order}, but got {zi.shape[-1]}"
            )

    if len(b) < order + 1:
        b = jnp.pad(b, (0, order + 1 - len(b)))
    if len(a) < order + 1:
        a = jnp.pad(a, (0, order + 1 - len(a)))

    b0 = b[0]
    b_ = b[None, 1:]
    a_ = a[None, 1:]

    def step_fn(carry, xn):
        yn = b0 * xn + carry[:, 0]
        carry = jnp.roll(carry, shift=-1, axis=1)
        carry = carry.at[:, -1].set(0)
        carry = carry + b_ * xn[:, None] - a_ * yn[:, None]
        return carry, yn

    in_shape = x.shape
    x_flat = x.reshape(-1, in_shape[-1]).T
    zi_flat = zi.reshape(-1, order)

    zi_final, y_flat = lax.scan(step_fn, zi_flat, x_flat)

    y = y_flat.T.reshape(in_shape)
    zi_final = zi_final.reshape(x.shape[:-1] + (order,))

    if return_zi:
        return y, zi_final
    else:
        return y
