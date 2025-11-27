from typing import overload, Literal

import jax.lax as lax
import jax.numpy as jnp


from jaxtyping import Float, Array


@overload
def lfilter(
    x: Float[Array, " n_samples"],
    a: Float[Array, " n_a"],
    b: Float[Array, " n_b"],
    zi: Float[Array, " order"] | None = None,
    *,
    return_zi: Literal[False] = False,
) -> Float[Array, " n_samples"]: ...


@overload
def lfilter(
    x: Float[Array, " n_samples"],
    a: Float[Array, " n_a"],
    b: Float[Array, " n_b"],
    zi: Float[Array, " order"] | None = None,
    *,
    return_zi: Literal[True],
) -> tuple[Float[Array, " n_samples"], Float[Array, " order"]]: ...


def lfilter(
    x: Float[Array, " n_samples"],
    a: Float[Array, " n_a"],
    b: Float[Array, " n_b"],
    zi: Float[Array, " order"] | None = None,
    *,
    return_zi: bool = False,
) -> (
    tuple[Float[Array, " n_samples"], Float[Array, " order"]]
    | Float[Array, " n_samples"]
):
    """Apply a time-invariant filter to the input signal.

    Filter is implemented in TDF2 structure. Coefficients will be zero padded to have the same length and will be normalized such that `a[0] = 1`.

    This function only operates on 1D signals, use `jax.vmap` to apply it to batched inputs.


    Args:
        x: Input signal of shape `(n_samples,)`.
        a: Denominator (IIR) coefficients of shape `(n_a,)`.
        b: Numerator (FIR) coefficients of shape `(n_b,)`.
        zi: Initial conditions of shape `(order,)`, where `order=max(n_a, n_b) - 1`. If `None`, zeros are used.
        return_zi: If `True`, return the final conditions along with the output.

    Returns:
        If `return_zi` is `False`, returns the filtered signal of shape `(n_samples,)`. If `return_zi` is `True`, returns a tuple containing:

            - Filtered signal of shape `(n_samples,)`
            - Final conditions of shape `(order,)`, where `order=max(n_a, n_b) - 1`.
    """
    b = b / a[0]
    a = a / a[0]

    order = max(a.shape[-1], b.shape[-1]) - 1

    if zi is None:
        zi = jnp.zeros((order,), dtype=x.dtype)
    else:
        assert zi.shape == (order,)

    if b.shape[-1] < order + 1:
        b = jnp.pad(b, (0, order + 1 - b.shape[-1]))
    if a.shape[-1] < order + 1:
        a = jnp.pad(a, (0, order + 1 - a.shape[-1]))

    def step_fn(carry, xn):
        yn = b[0] * xn + carry[0]
        carry = jnp.r_[carry[1:], jnp.array(0.0, dtype=carry.dtype)]
        carry = carry + b[1:] * xn - a[1:] * yn
        return carry, yn

    zi_out, y = lax.scan(step_fn, zi, x)

    if return_zi:
        return y, zi_out
    else:
        return y


def sosfilt(
    x: Float[Array, " n_samples"],
    a: Float[Array, " n_sections 3"],
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
        a: Denominator (IIR) coefficients of shape `(n_sections, 3)`.
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
