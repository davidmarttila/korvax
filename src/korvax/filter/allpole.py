from typing import overload, Literal
from functools import partial

import jax
import jax.lax as lax
import jax.numpy as jnp


from jaxtyping import Float, Array


@overload
def lfilter(
    x: Float[Array, " n_samples"],
    a: Float[Array, " n_a"],
    b: Float[Array, " n_b"],
    zi: None = None,
) -> Float[Array, " n_samples"]: ...


@overload
def lfilter(
    x: Float[Array, " n_samples"],
    a: Float[Array, " n_a"],
    b: Float[Array, " n_b"],
    zi: Float[Array, " order"],
) -> tuple[Float[Array, " n_samples"], Float[Array, " order"]]: ...


def lfilter(
    x: Float[Array, " n_samples"],
    a: Float[Array, " n_a"],
    b: Float[Array, " n_b"],
    zi: Float[Array, " order"] | None = None,
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

    zi, y = lax.scan(step_fn, zi, x)

    if return_zi:
        return y, zi
    else:
        return y


@overload
def allpole(
    x: Float[Array, " n_samples"],
    a: Float[Array, " n_samples order"],
    zi: Float[Array, " order"] | None = None,
    *,
    return_zi: Literal[False] = False,
) -> Float[Array, " n_samples"]: ...


@overload
def allpole(
    x: Float[Array, " n_samples"],
    a: Float[Array, " n_samples order"],
    zi: Float[Array, " order"] | None = None,
    *,
    return_zi: Literal[True],
) -> tuple[Float[Array, " n_samples"], Float[Array, " order"]]: ...


@partial(jax.custom_vjp, nondiff_argnames=("return_zi",))
def allpole(
    x: Float[Array, " n_samples"],
    a: Float[Array, " n_samples order"],
    zi: Float[Array, " order"] | None = None,
    return_zi: bool = False,
) -> (
    Float[Array, " n_samples"]
    | tuple[Float[Array, " n_samples"], Float[Array, " order"]]
):
    order = a.shape[-1]
    if zi is None:
        zi = jnp.zeros((order,), dtype=x.dtype)
    x = jnp.r_[zi, x]
    out = jax.ffi.ffi_call(
        "allpole", jax.ShapeDtypeStruct(x.shape, x.dtype), vmap_method="broadcast_all"
    )(x, a)

    if return_zi:
        return out[..., order:], out[..., -order:]
    return out[..., order:]


def allpole_fwd(x, a, zi, return_zi=False):
    y, zi_out = allpole(x, a, zi, return_zi=True)
    return (y, zi_out) if return_zi else y, (y, a, zi)


def allpole_bwd(return_zi, res, grad_y):
    y, a, zi = res
    n_samples, order = a.shape

    flipped_a = jnp.flip(a, axis=-1).T

    padded_a = jnp.pad(flipped_a, ((0, 0), (0, order + 1)))

    reshaped_a = jnp.reshape(padded_a, (n_samples + order + 1, order))
    sliced_a = reshaped_a[:-1, :]

    shifted_a = jnp.reshape(sliced_a, (order, n_samples + order)).T
    shifted_a = jnp.flip(shifted_a, axis=-1)

    if zi is None:
        shifted_a = shifted_a[order:]
        padded_grad_y = grad_y
    else:
        padded_grad_y = jnp.pad(grad_y, ((order, 0)), mode="constant")

    flipped_padded_grad_y = jnp.flip(padded_grad_y, axis=-1)
    flipped_shifted_a = jnp.flip(shifted_a, axis=0).conj()

    flipped_grad_x = allpole(flipped_padded_grad_y, flipped_shifted_a, zi)

    grad_zi = flipped_grad_x[-order:] if zi is not None else None
    flipped_grad_x = flipped_grad_x[:-order] if zi is not None else flipped_grad_x

    grad_x = jnp.flip(flipped_grad_x) if zi is not None else flipped_grad_x

    valid_y = y[:-1]
    padded_y = jnp.concatenate(
        [
            jnp.flip(zi) if zi is not None else jnp.zeros((order,), dtype=y.dtype),
            valid_y,
        ],
        axis=-1,
    )

    start_idxs = jnp.arange(padded_y.shape[0] - order + 1)
    unfolded_y = jax.vmap(
        partial(lax.dynamic_slice_in_dim, operand=padded_y, slice_size=order)
    )(start_index=start_idxs)

    grad_A = jnp.flip(unfolded_y, axis=1).conj() * -jnp.flip(flipped_grad_x)[:, None]
    return grad_x, grad_A, grad_zi


allpole.defvjp(allpole_fwd, allpole_bwd)  # pyright: ignore[reportFunctionMemberAccess]
