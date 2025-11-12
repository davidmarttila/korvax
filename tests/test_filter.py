import sys

import pytest
import jax
import jax.test_util
import jax.numpy as jnp
import numpy as np


import torch
from torchaudio.functional import lfilter as torch_lfilter
import korvax
from korvax.filter import lfilter as korvax_lfilter
from scipy.signal import lfilter as scipy_lfilter, sosfilt as scipy_sosfilt
from jaxpole.filter import allpole


@pytest.fixture
def x():
    return jax.random.normal(jax.random.key(0), (5, 1000))


@pytest.fixture
def a():
    return jnp.array([1.0, -0.95, 0.7, -0.3])


@pytest.fixture
def b():
    return jnp.array([0.3, 0.1, 0.2, 0.4])


def test_lfilter_output(b, a, x):
    y_korvax = korvax_lfilter(x, b=b, a=a, clamp=False)
    y_scipy = scipy_lfilter(
        np.array(b, dtype=np.float32),
        np.array(a, dtype=np.float32),
        np.array(x, dtype=np.float32),
    )
    y_torch = torch_lfilter(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(a, dtype=torch.float32),
        torch.tensor(b, dtype=torch.float32),
        clamp=False,
    ).numpy()

    assert jnp.allclose(y_korvax, jnp.array(y_scipy), atol=1e-5)
    assert jnp.allclose(y_korvax, y_torch, atol=1e-5)


@pytest.mark.parametrize("clamp", [True, False])
def test_lfilter_grads(b, a, x, clamp):
    korvax_grad_fn = jax.grad(
        lambda b, a, x: jnp.mean(korvax_lfilter(x, a=a, b=b, clamp=clamp) ** 2),
        argnums=(0, 1),
    )

    x_torch = torch.tensor(x, dtype=torch.float32)
    a_torch = torch.tensor(a, dtype=torch.float32, requires_grad=True)
    b_torch = torch.tensor(b, dtype=torch.float32, requires_grad=True)

    def torch_loss_fn(b, a, x):
        y = torch_lfilter(x, a, b, clamp=clamp)
        return torch.mean(y**2)

    korvax_grads = korvax_grad_fn(b, a, x)
    torch_loss = torch_loss_fn(b_torch, a_torch, x_torch)
    torch_loss.backward()
    torch_grads = np.asarray(b_torch.grad), np.asarray(a_torch.grad)

    assert jnp.allclose(korvax_grads[0], jnp.asarray(torch_grads[0]), atol=1e-5)
    assert jnp.allclose(korvax_grads[1], jnp.asarray(torch_grads[1]), atol=1e-5)


def test_time_varying_all_pole_output(a):
    b = 2
    order = 3

    x = jax.random.normal(jax.random.key(0), (b, 16000))
    a = a / a[0]

    y_static = korvax.filter.lfilter(x, a=a, b=jnp.array([1.0]), clamp=False)

    a = jnp.tile(a[1:, None], (1, x.shape[-1]))

    y_korvax = korvax.filter.time_varying_all_pole(x, a=a, clamp=False)
    a = a.swapaxes(0, 1)
    a = jnp.tile(a[None, :, :], (b, 1, 1))
    y_jaxpole = allpole(x, a, jnp.zeros((b, order)))

    assert jnp.allclose(y_korvax, y_static, atol=1e-5)
    assert jnp.allclose(y_korvax, y_jaxpole, atol=1e-5)


@pytest.mark.parametrize("time", [1, 2, 3])
def test_time_varying_all_pole_initial_grads_are_zero(a, time):
    b = 10

    x = jax.random.normal(jax.random.key(0), (b, time))
    a = a / a[0]
    a = jnp.tile(a[None, 1:, None], (b, 1, x.shape[-1]))

    grads = jax.grad(
        lambda a: jnp.mean(korvax.filter.time_varying_all_pole(x, a=a) ** 2)
    )(a)
    assert jnp.count_nonzero(grads[..., (time - 1) :]) == 0

    if time >= 2:
        assert jnp.count_nonzero(grads[..., : (time - 1)]) > 0

    from korvax.filter import time_varying_all_pole

    # type checking does not play nice with jax.test_util.check_grads
    tvap = time_varying_all_pole.__wrapped__  # pyright: ignore[reportFunctionMemberAccess]

    jax.test_util.check_grads(
        tvap,
        (x, a),
        order=2,
    )


@pytest.mark.skipif(sys.platform != "linux", reason="only check torchlpc on linux")
@pytest.mark.parametrize("order", [2, 4, 6])
def test_time_varying_all_pole_against_torchlpc(x, order):
    import torchlpc  # pyright: ignore[reportMissingImports]

    a = jax.random.normal(jax.random.key(1), x.shape + (order,))
    a = a.at[..., 0].set(1.0)

    y_korvax = korvax.filter.time_varying_all_pole(
        x, a=a.transpose(0, 2, 1), clamp=False
    )

    x_torch = torch.tensor(x, dtype=torch.float32)
    a_torch = torch.tensor(a, dtype=torch.float32, requires_grad=True)
    y_torch = torchlpc.sample_wise_lpc(x_torch, a_torch)

    assert jnp.allclose(y_korvax, y_torch.detach().numpy(), atol=1e-5)

    korvax_grads = jax.grad(
        lambda a: jnp.mean(
            korvax.filter.time_varying_all_pole(x, a=a, clamp=False) ** 2
        )
    )(a.transpose(0, 2, 1)).transpose(0, 2, 1)

    (torch.mean(y_torch**2)).backward()

    assert a_torch.grad is not None
    torch_grads = a_torch.grad.detach().numpy()

    assert jnp.allclose(korvax_grads, torch_grads, atol=1e-5)


def test_sosfilt_output(x):
    b0 = jnp.array([0.1, 0.2])
    b1 = jnp.array([0.2, 0.3])
    b2 = jnp.array([0.3, -0.1])
    a1 = jnp.array([-0.4, 0.4])
    a2 = jnp.array([0.5, -0.2])

    y_korvax = korvax.filter.sosfilt(x, b0=b0, b1=b1, b2=b2, a1=a1, a2=a2, clamp=False)

    sos = np.array(
        [[0.1, 0.2, 0.3, 1.0, -0.4, 0.5], [0.2, 0.3, -0.1, 1.0, 0.4, -0.2]],
        dtype=np.float32,
    )
    y_scipy = scipy_sosfilt(sos, np.array(x, dtype=np.float32))
    assert jnp.allclose(y_korvax, jnp.array(y_scipy), atol=1e-5)


@pytest.mark.parametrize("clamp", [True, False])
def test_time_varying_biquad_output(x, clamp):
    b = jnp.array([0.2, 0.3, 0.4])
    a = jnp.array([1.0, -0.5, 0.25])

    b_ = jnp.tile(b[:, None], (1, x.shape[-1]))
    a_ = jnp.tile(a[:, None], (1, x.shape[-1]))

    b0, b1, b2 = b_[0], b_[1], b_[2]
    a1, a2 = a_[1], a_[2]

    biquad_out = korvax.filter.time_varying_biquad(
        x,
        b0=b0,
        b1=b1,
        b2=b2,
        a1=a1,
        a2=a2,
        clamp=clamp,
    )

    lfilter_out = korvax.filter.lfilter(
        x,
        b=b,
        a=a,
        clamp=clamp,
    )

    assert jnp.allclose(biquad_out, lfilter_out, atol=1e-5)


def test_time_varying_sosfilt(x):
    b0 = jnp.array([0.1, 0.2])
    b1 = jnp.array([0.2, 0.3])
    b2 = jnp.array([0.3, -0.1])
    a1 = jnp.array([-0.4, 0.4])
    a2 = jnp.array([0.5, -0.2])

    b0_ = jnp.tile(b0[:, None], (1, x.shape[-1]))
    b1_ = jnp.tile(b1[:, None], (1, x.shape[-1]))
    b2_ = jnp.tile(b2[:, None], (1, x.shape[-1]))
    a1_ = jnp.tile(a1[:, None], (1, x.shape[-1]))
    a2_ = jnp.tile(a2[:, None], (1, x.shape[-1]))

    tv_sosfilt_out = korvax.filter.time_varying_sosfilt(
        x,
        b0=b0_,
        b1=b1_,
        b2=b2_,
        a1=a1_,
        a2=a2_,
        clamp=False,
    )

    sos = np.array(
        [[0.1, 0.2, 0.3, 1.0, -0.4, 0.5], [0.2, 0.3, -0.1, 1.0, 0.4, -0.2]],
        dtype=np.float32,
    )
    lfilter_out = scipy_sosfilt(sos, np.array(x, dtype=np.float32))

    sosfilt_out = korvax.filter.sosfilt(
        x,
        b0=b0,
        b1=b1,
        b2=b2,
        a1=a1,
        a2=a2,
        clamp=False,
    )

    assert jnp.allclose(tv_sosfilt_out, jnp.array(lfilter_out), atol=1e-5)
    assert jnp.allclose(tv_sosfilt_out, sosfilt_out, atol=1e-5)
