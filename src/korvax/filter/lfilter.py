from typing import overload

import jax
import jax.lax as lax
import jax.numpy as jnp


from jaxtyping import Float, Array


@overload
def lfilter(
    x: Float[Array, "... n_samples"],
    a: Float[Array, "... n_a"],
    b: Float[Array, "... n_b"],
    zi: None = None,
) -> Float[Array, "... n_samples"]: ...


@overload
def lfilter(
    x: Float[Array, "... n_samples"],
    a: Float[Array, "... n_a"],
    b: Float[Array, "... n_b"],
    zi: Float[Array, "... order"],
) -> tuple[Float[Array, "... n_samples"], Float[Array, "... order"]]: ...


def lfilter(
    x: Float[Array, "... n_samples"],
    a: Float[Array, "... n_a"],
    b: Float[Array, "... n_b"],
    zi: Float[Array, "... order"] | None = None,
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

    if return_zi:
        return y, zi_final
    else:
        return y


@overload
def time_varying_all_pole(
    x: Float[Array, "... n_samples"],
    a: Float[Array, "... order n_samples"],
    zi: None = None,
) -> Float[Array, "... n_samples"]: ...


@overload
def time_varying_all_pole(
    x: Float[Array, "... n_samples"],
    a: Float[Array, "... order n_samples"],
    zi: Float[Array, "... order"],
) -> tuple[Float[Array, "... n_samples"], Float[Array, "... order"]]: ...


# @jax.custom_vjp
def time_varying_all_pole(
    x: Float[Array, "... n_samples"],
    a: Float[Array, "... order n_samples"],
    zi: Float[Array, "... order"] | None = None,
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
        # yn = xn - jnp.sum(an * carry, axis=-1)
        yn = xn - jnp.einsum("bo,bo->b", an, carry)
        carry = jnp.roll(carry, shift=1, axis=-1)
        carry = carry.at[:, 0].set(yn)
        return carry, yn

    in_shape = x.shape
    x_flat = x.reshape(-1, in_shape[-1]).T
    zi_flat = zi.reshape(-1, order)
    a_flat = a.reshape(-1, order, n_samples).transpose(2, 0, 1)

    zi_final, y_flat = lax.scan(step_fn, zi_flat, (x_flat, a_flat))

    y = y_flat.T.reshape(in_shape)
    zi_final = zi_final.reshape(x.shape[:-1] + (order,))
    # zi_final = y[..., -order:]

    if return_zi:
        return y, zi_final
    else:
        return y


def tvap_fwd(x, a, zi):
    if zi is None:
        y = time_varying_all_pole(x, a, zi=None)
        out = y
    else:
        y, zi_out = time_varying_all_pole(x, a, zi)
        out = (y, zi_out)

    return out, (y, a, zi)


def tvap_bwd(res, grad_y):
    y, a, zi = res
    *batch_shape, order, n_samples = a.shape

    flipped_a = jnp.flip(a, axis=-2)

    padded_a = jnp.pad(flipped_a, ((0, 0),) * (flipped_a.ndim - 1) + ((0, order + 1),))

    reshaped_a = jnp.reshape(padded_a, (*batch_shape, n_samples + order + 1, order))
    sliced_a = reshaped_a[..., :-1, :]

    shifted_a = jnp.reshape(sliced_a, (*batch_shape, order, n_samples + order))
    shifted_a = jnp.flip(shifted_a, axis=-2)

    if zi is None:
        shifted_a = shifted_a[..., order:]
        padded_grad_y = grad_y
    else:
        padded_grad_y = jnp.pad(grad_y[0], ((0, 0), (order, 0)), mode="constant")

    flipped_padded_grad_y = jnp.flip(padded_grad_y, axis=-1)
    flipped_shifted_a = jnp.flip(shifted_a, axis=-1).conj()

    flipped_grad_x = time_varying_all_pole(
        flipped_padded_grad_y, flipped_shifted_a, zi=zi
    )

    grad_zi = flipped_grad_x[..., -order:] if zi is not None else None
    flipped_grad_x = flipped_grad_x[..., :-order] if zi is not None else flipped_grad_x

    grad_x = jnp.flip(flipped_grad_x, axis=-1) if zi is not None else flipped_grad_x

    valid_y = y[..., :-1]
    padded_y = jnp.concatenate(
        [
            jnp.flip(zi, axis=-1)
            if zi is not None
            else jnp.zeros((*batch_shape, order), dtype=y.dtype),
            valid_y,
        ],
        axis=-1,
    )

    start_idxs = jnp.arange(padded_y.shape[1] - order + 1)
    unfolded_y = jax.vmap(
        lax.dynamic_slice_in_dim, in_axes=(None, 0, None, None), out_axes=-1
    )(
        padded_y,
        start_idxs,
        order,
        -1,
    )

    grad_A = (
        jnp.flip(unfolded_y, axis=-2).conj()
        * -jnp.flip(flipped_grad_x, axis=-1)[..., None, :]
    )
    return grad_x, grad_A, grad_zi


# time_varying_all_pole.defvjp(tvap_fwd, tvap_bwd)
