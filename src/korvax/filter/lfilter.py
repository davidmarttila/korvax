from typing import overload

import jax.lax as lax
import jax.numpy as jnp

from jaxtyping import Float, Array


@overload
def lfilter(
    x: Float[Array, "... n_samples"],
    /,
    a: Float[Array, "... n_a"],
    b: Float[Array, "... n_b"],
    zi: None = None,
    clamp: bool = True,
) -> Float[Array, "... n_samples"]: ...


@overload
def lfilter(
    x: Float[Array, "... n_samples"],
    /,
    a: Float[Array, "... n_a"],
    b: Float[Array, "... n_b"],
    zi: Float[Array, "... order"],
    clamp: bool = True,
) -> tuple[Float[Array, "... n_samples"], Float[Array, "... order"]]: ...


def lfilter(
    x: Float[Array, "... n_samples"],
    /,
    a: Float[Array, "... n_a"],
    b: Float[Array, "... n_b"],
    zi: Float[Array, "... order"] | None = None,
    clamp: bool = True,
) -> (
    tuple[Float[Array, "... n_samples"], Float[Array, "... order"]]
    | Float[Array, "... n_samples"]
):
    b = b / a[..., 0:1]
    a = a / a[..., 0:1]

    order = max(a.shape[-1], b.shape[-1]) - 1
    if zi is None:
        return_zi = False
        zi = jnp.zeros(x.shape[:-1] + (order,), dtype=x.dtype)
    else:
        return_zi = True
        if zi.shape[-1] != order:
            raise ValueError(
                f"Initial conditions zi must have length {order}, but got {zi.shape[-1]}"
            )

    if b.shape[-1] < order + 1:
        b = jnp.pad(b, ((0, 0),) * (b.ndim - 1) + ((0, order + 1 - b.shape[-1]),))
    if a.shape[-1] < order + 1:
        a = jnp.pad(a, ((0, 0),) * (a.ndim - 1) + ((0, order + 1 - a.shape[-1]),))

    b0 = b[..., 0].reshape(-1)
    b_ = b[..., 1:].reshape(-1, order)
    a_ = a[..., 1:].reshape(-1, order)

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

    if clamp:
        y = y.clip(-1.0, 1.0)

    if return_zi:
        return y, zi_final
    else:
        return y


@overload
def time_varying_all_pole(
    x: Float[Array, "... n_samples"],
    /,
    a: Float[Array, "... order n_samples"],
    zi: None = None,
    clamp: bool = True,
) -> Float[Array, "... n_samples"]: ...


@overload
def time_varying_all_pole(
    x: Float[Array, "... n_samples"],
    /,
    a: Float[Array, "... order n_samples"],
    zi: Float[Array, "... order"],
    clamp: bool = True,
) -> tuple[Float[Array, "... n_samples"], Float[Array, "... order"]]: ...


def time_varying_all_pole(
    x: Float[Array, "... n_samples"],
    /,
    a: Float[Array, "... order n_samples"],
    zi: Float[Array, "... order"] | None = None,
    clamp: bool = True,
) -> (
    tuple[Float[Array, "... n_samples"], Float[Array, "... order"]]
    | Float[Array, "... n_samples"]
):
    n_samples = x.shape[-1]
    order = a.shape[-2]
    if zi is None:
        return_zi = False
        zi = jnp.zeros(x.shape[:-1] + (order,), dtype=x.dtype)
    else:
        return_zi = True
        if zi.shape[-1] != order:
            raise ValueError(
                f"Initial conditions zi must have length {order}, but got {zi.shape[-1]}"
            )

    def step_fn(carry, inputs):
        xn, an = inputs
        yn = jnp.sum(-an * carry, axis=-1) + xn
        carry = jnp.concat([yn[:, None], carry[:, :-1]], axis=-1)
        return carry, yn

    in_shape = x.shape
    x_flat = x.reshape(-1, in_shape[-1]).T
    zi_flat = zi.reshape(-1, order)
    a_flat = a.reshape(-1, order, n_samples).transpose(2, 0, 1)

    zi_final, y_flat = lax.scan(step_fn, zi_flat, (x_flat, a_flat))

    y = y_flat.T.reshape(in_shape)
    zi_final = zi_final.reshape(x.shape[:-1] + (order,))

    if clamp:
        y = y.clip(-1.0, 1.0)

    if return_zi:
        return y, zi_final
    else:
        return y
