from typing import overload, Literal

import jax
import jax.numpy as jnp
import jax.lax as lax

from jaxtyping import Float, Array


def _interleave(a, b):
    a_pad = [(0, 0, 0)] * a.ndim
    b_pad = [(0, 0, 0)] * b.ndim
    a_pad[0] = (0, 1 if a.shape[0] == b.shape[0] else 0, 1)
    b_pad[0] = (1, 0 if a.shape[0] == b.shape[0] else 1, 1)
    return lax.add(
        lax.pad(a, jnp.array(0, dtype=a.dtype), a_pad),
        lax.pad(b, jnp.array(0, dtype=b.dtype), b_pad),
    )


@jax.custom_vjp
def _recurrence(A, z):
    def _scan(elems, A_):
        num_elems = elems.shape[0]
        if num_elems < 2:
            return elems

        reduced_elems = jnp.matvec(A_, elems[0:-1:2]) + elems[1::2]

        odd_elems = _scan(reduced_elems, A_ @ A_)
        if num_elems % 2 == 0:
            even_elems = jnp.matvec(A_, odd_elems[0:-1]) + elems[2::2]
        else:
            even_elems = jnp.matvec(A_, odd_elems) + elems[2::2]

        even_elems = jnp.concatenate([elems[0:1, ...], even_elems], axis=0)
        return _interleave(even_elems, odd_elems)

    return _scan(z, A)


def _recurrence_fwd(A, z):
    v = _recurrence(A, z)
    return v, (A, v)


def _recurrence_bwd(res, g):
    A, v = res

    A = A.T.conj()

    grad_z = jnp.flip(_recurrence(A, jnp.flip(g, axis=0)), axis=0)
    grad_A = grad_z[1:, ...].T @ v[:-1, ...].conj()

    return grad_A, grad_z


_recurrence.defvjp(_recurrence_fwd, _recurrence_bwd)


def _companion(a):
    M = a.shape[-1]
    C = jnp.zeros((M, M), dtype=a.dtype)
    C = C.at[jnp.arange(1, M), jnp.arange(M - 1)].set(1.0)
    C = C.at[0, :].set(-a)
    return C


@overload
def lfilter(
    x: Float[Array, " n_samples"],
    a: Float[Array, " n_a"] | None = None,
    b: Float[Array, " n_b"] | None = None,
    zi: Float[Array, " order"] | None = None,
    *,
    return_zi: Literal[False] = False,
    transposed: bool = True,
) -> Float[Array, " n_samples"]: ...


@overload
def lfilter(
    x: Float[Array, " n_samples"],
    a: Float[Array, " n_a"] | None = None,
    b: Float[Array, " n_b"] | None = None,
    zi: Float[Array, " order"] | None = None,
    *,
    return_zi: Literal[True],
    transposed: bool = True,
) -> tuple[Float[Array, " n_samples"], Float[Array, " order"]]: ...


def lfilter(
    x: Float[Array, " n_samples"],
    a: Float[Array, " n_a"] | None = None,
    b: Float[Array, " n_b"] | None = None,
    zi: Float[Array, " order"] | None = None,
    *,
    return_zi: bool = False,
    transposed: bool = True,
) -> (
    tuple[Float[Array, " n_samples"], Float[Array, " order"]]
    | Float[Array, " n_samples"]
):
    """Apply a time-invariant filter to the input signal.

    Filter is implemented in TDF2 structure. Coefficients will be zero padded to have the same length and will be normalized such that `a[0] = 1`.

    This function only operates on 1D signals, use `jax.vmap` to apply it to batched inputs.


    Args:
        x: Input signal of shape `(n_samples,)`.
        a: Denominator (IIR) coefficients `a_1, a_2, ...` of shape `(n_a)`. `a_0 = 1` is implied. Can be `None` for all-zero filters.
        b: Numerator (FIR) coefficients `b_0, b_1, ...` of shape `(n_b)`. Can be `None` for all-pole filters.
        zi: Initial conditions of shape `(order,)`, where `order=max(n_a, n_b) - 1`. If `None`, zeros are used.
        return_zi: If `True`, return the final conditions along with the output.
        transposed: Whether to use transposed direct form II structure (default). Uses direct form II otherwise.

    Returns:
        If `return_zi` is `False`, returns the filtered signal of shape `(n_samples,)`. If `return_zi` is `True`, returns a tuple containing:

            - Filtered signal of shape `(n_samples,)`
            - Final conditions of shape `(order,)`, where `order=max(n_a, n_b) - 1`.
    """
    if a is None:
        a = jnp.array([], dtype=x.dtype)

    if b is None:
        b = jnp.array([1.0], dtype=x.dtype)

    order = max(a.shape[-1], b.shape[-1] - 1)

    if zi is None:
        zi = jnp.zeros((order,), dtype=x.dtype)
    else:
        assert zi.shape == (order,)

    assert zi is not None

    if b.shape[-1] < order + 1:
        b = jnp.pad(b, (0, order + 1 - b.shape[-1]))
    if a.shape[-1] < order:
        a = jnp.pad(a, (0, order - a.shape[-1]))

    A = _companion(a)
    b0 = b[:1]
    C = b[1:] - a * b0
    D = b[0]

    if transposed:
        A = A.T
        z = x[:, None] * C

    else:
        z = jnp.pad(x[:, None], ((0, 0), (0, order - 1)))

    v = _recurrence(A, jnp.concatenate([zi[None, :], z], axis=0))

    if transposed:
        y = v[:-1, 0] + D * x
    else:
        y = jnp.dot(v[:-1, :], C) + D * x

    if return_zi:
        return y, v[-1, :]
    else:
        return y


@overload
def sosfilt(
    x: Float[Array, " n_samples"],
    a: Float[Array, " n_sections 2"],
    b: Float[Array, " n_sections 3"],
    zi: Float[Array, " n_sections 2"] | None = None,
    *,
    return_zi: Literal[False] = False,
) -> Float[Array, " n_samples"]: ...


@overload
def sosfilt(
    x: Float[Array, " n_samples"],
    a: Float[Array, " n_sections 2"],
    b: Float[Array, " n_sections 3"],
    zi: Float[Array, " n_sections 2"] | None = None,
    *,
    return_zi: Literal[True],
) -> tuple[Float[Array, " n_samples"], Float[Array, " n_sections 2"]]: ...


def sosfilt(
    x: Float[Array, " n_samples"],
    a: Float[Array, " n_sections 2"],
    b: Float[Array, " n_sections 3"],
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
        a: Denominator (IIR) coefficients of shape `(n_sections, 2)`.
        b: Numerator (FIR) coefficients of shape `(n_sections, 3)`.
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
