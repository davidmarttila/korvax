from typing import overload

from functools import partial
import jax
import jax.lax as lax
import jax.numpy as jnp

from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.experimental import pallas as pl

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


@jax.custom_vjp
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

    in_shape = x.shape

    x_flat = x.reshape(-1, in_shape[-1])
    a_flat = a.reshape(-1, order, n_samples).transpose(0, 2, 1)
    zi_flat = zi.reshape(-1, order)

    batch_size = x_flat.shape[0]

    if jax.default_backend() == "gpu":

        @partial(
            plgpu.kernel,
            out_shape=x_flat,
            scratch_shapes=dict(
                sm_ref=plgpu.SMEM(shape=(batch_size, order), dtype=x.dtype),
                barrier_ref=plgpu.Barrier(num_arrivals=order, num_barriers=batch_size),
            ),
            grid=(batch_size,),
            grid_names=("batch",),
            num_threads=order,
            thread_name="order",
        )
        def run_kernel(x_ref, a_ref, zi_ref, y_ref, sm_ref, barrier_ref):
            i = lax.axis_index("order")
            b = lax.axis_index("batch")

            sm_ref[b, i] = zi_ref[b, order - i - 1]
            plgpu.barrier_wait(barrier_ref.at[b])

            def _step(t, _):
                circular_idx = lax.rem(t, order)
                a = -a_ref[b, t, i]

                sm_idx = lax.cond(
                    i > circular_idx - 1,
                    lambda: circular_idx - i - 1 + order,
                    lambda: circular_idx - i - 1,
                )

                v = a * sm_ref[b, sm_idx]

                @pl.when(i == order - 1)
                def update_state():
                    sm_ref[b, circular_idx] = v

                v = lax.cond(
                    i == order - 1,
                    lambda: v,
                    lambda: x_ref[b, t],
                )

                plgpu.barrier_wait(barrier_ref.at[b])
                pl.atomic_add(sm_ref, (b, circular_idx), v)
                plgpu.barrier_wait(barrier_ref.at[b])

                @pl.when(i == order - 1)
                def update_output():
                    y_ref[b, t] = sm_ref[b, circular_idx]

            lax.fori_loop(0, n_samples, _step, None)

    else:

        def _run_kernel(x_, a_, zi_):
            def step(carry, inputs):
                xn, an = inputs
                yn = xn - jnp.sum(an * carry)
                carry = jnp.roll(carry, shift=1)
                return carry.at[0].set(yn), yn

            _, y = lax.scan(step, zi_, (x_, a_))
            return y

        run_kernel = jax.vmap(_run_kernel)  # pyright: ignore[reportAssignmentType]

    y_flat = run_kernel(x_flat, a_flat, zi_flat)
    y = y_flat.reshape(in_shape)

    if return_zi:
        return y, y[..., -order:]
    else:
        return y


def _tvap_fwd(x, a, zi):
    if zi is None:
        y = time_varying_all_pole(x, a, zi=None)
        out = y
    else:
        y, zi_out = time_varying_all_pole(x, a, zi)
        out = (y, zi_out)

    return out, (y, a, zi)


def _tvap_bwd(res, grad_y):
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


time_varying_all_pole.defvjp(_tvap_fwd, _tvap_bwd)  # pyright: ignore[reportFunctionMemberAccess]
