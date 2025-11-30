from functools import partial
from typing import overload, Literal

import jax
import jax.numpy as jnp
import jax.lax as lax

from jaxtyping import Float, Array


@jax.custom_vjp
def _recurrence(A, z, v0):
    def fn(t1, t2):
        a1, z1 = t1
        a2, z2 = t2
        return a2 @ a1, jnp.matvec(a2, z1) + z2

    order = A.shape[-1]
    A = jnp.concatenate([jnp.zeros((1, order, order)), A], axis=0)
    z = jnp.concatenate([v0[None, :], z], axis=0)

    _, v = lax.associative_scan(fn, (A, z))
    return v


def _recurrence_fwd(A, z, v0):
    v = _recurrence(A, z, v0)
    return v, (A, v, v0)


def _recurrence_bwd(res, g):
    A, v, v0 = res

    AmT = A.mT.conj()
    AmT_rolled = jnp.roll(AmT, shift=-1, axis=0)
    g = jnp.flip(g[:-1], axis=0)

    grad_z = jnp.flip(
        _recurrence(jnp.flip(AmT_rolled, axis=0), g, jnp.zeros_like(v0))[:-1], axis=0
    )
    grad_v0 = jnp.matvec(AmT[0], grad_z[0, :])
    grad_A = v[:-1].conj()[:, None, :] * grad_z[:, :, None]

    return grad_A, grad_z, grad_v0


_recurrence.defvjp(_recurrence_fwd, _recurrence_bwd)  # pyright: ignore[reportFunctionMemberAccess]


def _companion(a):
    T, M = a.shape
    C = jnp.zeros((M, M), dtype=a.dtype)
    C = C.at[jnp.arange(1, M), jnp.arange(M - 1)].set(1.0)

    C = jnp.tile(C[None, :, :], (T, 1, 1))
    C = C.at[:, 0, :].set(-a)
    return C


@overload
def lfilter(
    x: Float[Array, " n_samples"],
    a: Float[Array, " n_samples n_a"] | None = None,
    b: Float[Array, " n_samples n_b"] | None = None,
    zi: Float[Array, " order"] | None = None,
    *,
    return_zi: Literal[False] = False,
    transposed: bool = False,
) -> Float[Array, " n_samples"]: ...


@overload
def lfilter(
    x: Float[Array, " n_samples"],
    a: Float[Array, " n_samples n_a"] | None = None,
    b: Float[Array, " n_samples n_b"] | None = None,
    zi: Float[Array, " order"] | None = None,
    *,
    return_zi: Literal[True],
    transposed: bool = False,
) -> tuple[Float[Array, " n_samples"], Float[Array, " order"]]: ...


def lfilter(
    x: Float[Array, " n_samples"],
    a: Float[Array, " n_samples n_a"] | None = None,
    b: Float[Array, " n_samples n_b"] | None = None,
    zi: Float[Array, " order"] | None = None,
    *,
    return_zi: bool = False,
    transposed: bool = False,
) -> (
    tuple[Float[Array, " n_samples"], Float[Array, " order"]]
    | Float[Array, " n_samples"]
):
    """Apply a time-invariant filter to the input signal.

    Filtering is implemented using the state-space implementations with parallel associative scans as described in [1].
    No diagonalization is implemented currently! For filters with order > ~4, combining [korvax.filter.lpv.fir] and [korvax.filter.lpv.allpole] will likely be a lot more performant.

    This function only operates on 1D signals, use `jax.vmap` to apply it to batched inputs.


    Args:
        x: Input signal of shape `(n_samples,)`.
        a: Time-varying denominator (IIR) coefficients `a_1, a_2, ...` of shape `(n_samples, n_a)`. `a_0 = 1` is implied. Can be `None` for all-zero filters.
        b: Time-varying numerator (FIR) coefficients `b_0, b_1, ...` of shape `(n_samples, n_b)`. Can be `None` for all-pole filters.
        zi: Initial conditions of shape `(order,)`, where `order=max(n_a, n_b - 1)`. If `None`, zeros are used.
        return_zi: If `True`, return the final conditions along with the output.
        transposed: Whether to use transposed direct form II structure. Uses direct form II if False (default).

    Returns:
        If `return_zi` is `False`, returns the filtered signal of shape `(n_samples,)`. If `return_zi` is `True`, returns a tuple containing:

            - Filtered signal of shape `(n_samples,)`
            - Final conditions of shape `(order,)`, where `order=max(n_a, n_b) - 1`.

    References:
        [1] C.-Y. Yu and G. Fazekas. "Accelerating Automatic Differentiation of Direct Form Digital Filters", DiffSys Workshap at EurIPS, 2025.
    """
    n_samples = x.shape[0]

    if a is None:
        a = jnp.empty((n_samples, 0), dtype=x.dtype)

    if b is None:
        b = jnp.ones((n_samples, 1), dtype=x.dtype)

    order = max(a.shape[-1], b.shape[-1] - 1)

    if zi is None:
        zi = jnp.zeros((order,), dtype=x.dtype)
    else:
        assert zi.shape == (order,)

    assert zi is not None

    if b.shape[-1] < order + 1:
        b = jnp.pad(b, ((0, 0), (0, order + 1 - b.shape[-1])))
    if a.shape[-1] < order:
        a = jnp.pad(a, ((0, 0), (0, order - a.shape[-1])))

    A = _companion(a)
    b0 = b[:, :1]
    C = b[:, 1:] - a * b0
    D = b[:, 0]

    if transposed:
        A = A.mT
        z = x[:, None] * C

    else:
        z = jnp.pad(x[:, None], ((0, 0), (0, order - 1)))

    v = _recurrence(A, z, zi)

    if transposed:
        y = v[:-1, 0] + D * x
    else:
        y = jnp.linalg.vecdot(v[:-1, :], C) + D * x

    if return_zi:
        return y, v[-1, :]
    else:
        return y


@overload
def sosfilt(
    x: Float[Array, " n_samples"],
    a: Float[Array, " n_sections n_samples 2"],
    b: Float[Array, " n_sections n_samples 3"],
    zi: Float[Array, " n_sections 2"] | None = None,
    *,
    return_zi: Literal[False] = False,
) -> Float[Array, " n_samples"]: ...


@overload
def sosfilt(
    x: Float[Array, " n_samples"],
    a: Float[Array, " n_sections n_samples 2"],
    b: Float[Array, " n_sections n_samples 3"],
    zi: Float[Array, " n_sections 2"] | None = None,
    *,
    return_zi: Literal[True],
) -> tuple[Float[Array, " n_samples"], Float[Array, " n_sections 2"]]: ...


def sosfilt(
    x: Float[Array, " n_samples"],
    a: Float[Array, " n_sections n_samples 2"],
    b: Float[Array, " n_sections n_samples 3"],
    zi: Float[Array, " n_sections 2"] | None = None,
    *,
    return_zi: bool = False,
) -> (
    tuple[Float[Array, " n_samples"], Float[Array, " n_sections 2"]]
    | Float[Array, " n_samples"]
):
    """Apply a cascade of time-invariant second-order filters (biquads) to the input signal.

    This function only operates on 1D signals, use `jax.vmap` to apply it to batched inputs.


    Args:
        x: Input signal of shape `(n_samples,)`.
        a: Denominator (IIR) coefficients of shape `(n_sections, n_samples, 2)`.
        b: Numerator (FIR) coefficients of shape `(n_sections, n_samples, 3)`.
        zi: Initial conditions of shape `(n_sections, 2)`. If `None`, zeros are used.
        return_zi: If `True`, return the final conditions along with the output.

    Returns:
        If `return_zi` is `False`, returns the filtered signal of shape `(n_samples,)`. If `return_zi` is `True`, returns a tuple containing:
            - Filtered signal of shape `(n_samples,)`
            - Final conditions of shape `(n_sections, 2)`.
    """
    n_sections = a.shape[0]

    if zi is None:
        zi = jnp.zeros((n_sections, 2), dtype=x.dtype)
    else:
        assert zi.shape == (n_sections, 2)

    def _section(carry, inputs):
        a_, b_, zi_ = inputs
        return lfilter(carry, a_, b_, zi=zi_, return_zi=True)

    y, zi_out = lax.scan(_section, x, (a, b, zi))
    if return_zi:
        return y, zi_out
    else:
        return y


def fir(
    x: Float[Array, " n_samples"],
    b: Float[Array, " n_samples n_b"],
    zi: Float[Array, " n_b-1"] | None = None,
) -> Float[Array, " n_samples"]:
    """Apply a time-varying FIR filter to the input signal.

    This function only operates on 1D signals, use `jax.vmap` to apply it to batched inputs.

    Args:
        x: Input signal of shape `(n_samples,)`.
        b: Time-varying FIR coefficients `b_0, b_1, ...` of shape `(n_samples, order+1)`.
        zi: Initial conditions of shape `(order,)`. If `None`, zeros are used.

    Returns:
        Filtered signal of shape `(n_samples,)`.
    """
    order = b.shape[-1] - 1
    if zi is None:
        zi = jnp.zeros((order,), dtype=x.dtype)

    n_samples = x.shape[0]
    x = jnp.r_[zi, x]

    frames = jax.vmap(
        partial(lax.dynamic_slice_in_dim, operand=x, slice_size=order + 1)
    )(start_index=jnp.arange(n_samples))

    frames = jnp.flip(frames, axis=1)

    return jnp.linalg.vecdot(frames, b)


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
    """Apply a time-varying all-pole filter to the input signal.

    Port of `torchlpc.sample_wise_lpc`. Uses the efficient differentiation method proposed in [1].

    This function only operates on 1D signals, use `jax.vmap` to apply it to batched inputs.


    Args:
        x: Input signal of shape `(n_samples,)`.
        a: Time-varying all-pole coefficients of shape `(n_samples, order)`.
        zi: Initial conditions of shape `(order,)`. If `None`, zeros are used.
        return_zi: If `True`, return the final conditions along with the output.

    Returns:
        If `return_zi` is `False`, returns the filtered signal of shape `(n_samples,)`. If `return_zi` is `True`, returns a tuple containing:

            - Filtered signal of shape `(n_samples,)`
            - Final conditions of shape `(order,)`

    References:
        [1] C.-Y. Yu, C. Mitcheltree, A. Carson, S. Bilbao, J. D. Reiss, and G. Fazekas. "Differentiable All-Pole Filters for Time-Varying Audio Systems," in Proc. DAFx, 2024.
    """

    order = a.shape[-1]
    if zi is None:
        zi = jnp.zeros((order,), dtype=x.dtype)
    x = jnp.r_[zi, x]

    def _call(target: str):
        return jax.ffi.ffi_call(
            target, jax.ShapeDtypeStruct(x.shape, x.dtype), vmap_method="broadcast_all"
        )

    out = lax.platform_dependent(
        x,
        a,
        default=_call("allpole_cpu"),
        cpu=_call("allpole_cpu"),
        cuda=_call("allpole_cuda"),
    )

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
