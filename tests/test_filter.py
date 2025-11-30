from functools import partial
import pytest
import jax
import jax.numpy as jnp
import numpy as np

import torchlpc
import torch

from torchaudio.functional import lfilter as torch_lfilter
import korvax

from korvax.filter.lti import lfilter as korvax_lti_lfilter
import scipy.signal
import philtorch.lti
import philtorch.lpv


@pytest.fixture
def x():
    return jax.random.normal(jax.random.key(0), (5, 1000))


@pytest.fixture
def a():
    return jnp.array([1.0, -0.95, 0.7, -0.3])


@pytest.fixture
def b():
    return jnp.array([0.3, 0.1, 0.2, 0.4])


@pytest.mark.parametrize("transposed", [True, False])
def test_lti_lfilter_output(b, a, x, transposed):
    fn = partial(korvax_lti_lfilter, a=a[1:], b=b, transposed=transposed)

    y_korvax = jax.vmap(fn)(x=x)
    y_scipy = scipy.signal.lfilter(
        np.array(b, dtype=np.float32),
        np.array(a, dtype=np.float32),
        np.array(x, dtype=np.float32),
    )
    y_torch = philtorch.lti.lfilter(
        x=torch.tensor(x, dtype=torch.float32),
        a=torch.tensor(a[1:], dtype=torch.float32),
        b=torch.tensor(b, dtype=torch.float32),
    ).numpy()  # pyright: ignore[reportAttributeAccessIssue]

    assert jnp.allclose(y_korvax, jnp.array(y_scipy), atol=1e-5)
    assert jnp.allclose(y_korvax, y_torch, atol=1e-5)


def test_lti_lfilter_grads(b, a, x):
    korvax_grad_fn = jax.grad(
        lambda x, a, b: jnp.mean(
            jax.vmap(korvax_lti_lfilter, in_axes=(0, None, None))(x, a, b) ** 2
        ),
        argnums=(1, 2),
    )

    x_torch = torch.tensor(x, dtype=torch.float32)
    a_torch = torch.tensor(a, dtype=torch.float32, requires_grad=True)
    b_torch = torch.tensor(b, dtype=torch.float32, requires_grad=True)

    def torch_loss_fn(x, a, b):
        y = torch_lfilter(x, a, b, clamp=False)
        return torch.mean(y**2)

    korvax_grads = korvax_grad_fn(x, a[1:], b)
    torch_loss = torch_loss_fn(x_torch, a_torch, b_torch)
    torch_loss.backward()
    torch_grads = np.asarray(a_torch.grad)[1:], np.asarray(b_torch.grad)

    assert jnp.allclose(korvax_grads[0], jnp.asarray(torch_grads[0]), atol=1e-5)
    assert jnp.allclose(korvax_grads[1], jnp.asarray(torch_grads[1]), atol=1e-5)


def test_sosfilt_output(x, a, b):
    a = a[:3]
    b = b[:3]
    n_filt = 3

    a = jnp.tile(a[None, :], (n_filt, 1))
    b = jnp.tile(b[None, :], (n_filt, 1))

    np_sos = np.array(jnp.concatenate([b, a], axis=-1))

    y_korvax = jax.vmap(korvax.filter.lti.sosfilt, in_axes=(0, None, None))(
        x, a[:, 1:], b
    )
    y_scipy = scipy.signal.sosfilt(np_sos, np.array(x))

    assert jnp.allclose(y_korvax, y_scipy, atol=1e-5)  # pyright: ignore[reportArgumentType]


def test_ltv_lti_equivalence(x, a, b):
    n_samples = x.shape[1]

    a_ltv = jnp.tile(a[None, 1:], (n_samples, 1))
    b_ltv = jnp.tile(b[None, :], (n_samples, 1))

    y_korvax = jax.vmap(korvax.filter.ltv.lfilter, in_axes=(0, None, None))(
        x, a_ltv, b_ltv
    )

    y_lti = jax.vmap(korvax.filter.lti.lfilter, in_axes=(0, None, None))(x, a[1:], b)

    assert jnp.allclose(y_korvax, y_lti, atol=1e-5)


@pytest.mark.parametrize("order", [1, 2, 4, 6])
def test_ltv_against_torchlpc_values(x, order):
    a = jax.random.normal(jax.random.key(1), x.shape + (order,)) * 0.1

    y_korvax = jax.vmap(korvax.filter.ltv.lfilter)(x, a=a)

    x_torch = torch.tensor(np.array(x), dtype=torch.float32)
    a_torch = torch.tensor(np.array(a), dtype=torch.float32, requires_grad=True)
    y_torch = torchlpc.sample_wise_lpc(x_torch, a_torch)

    assert jnp.allclose(y_korvax, y_torch.detach().numpy(), atol=1e-5)  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.parametrize("order", [1, 2, 4, 6])
def test_ltv_grads(x, order):
    a = jax.random.normal(jax.random.key(1), x.shape + (order,)) * 0.1
    # a = jnp.tile(a[:, 0:1, :], (1, x.shape[1], 1))

    x_torch = torch.tensor(np.array(x), dtype=torch.float32)
    a_torch = torch.tensor(np.array(a), dtype=torch.float32, requires_grad=True)
    b_torch = torch.zeros(x.shape + (order + 1,), dtype=torch.float32)
    b_torch[..., 0] = 1.0
    y_torch = philtorch.lpv.lfilter(x=x_torch, a=a_torch, b=b_torch)

    korvax_grads = jax.grad(
        lambda a: jnp.mean(jax.vmap(korvax.filter.ltv.lfilter)(x, a=a) ** 2)
    )(a)

    (torch.mean(y_torch**2)).backward()  # pyright: ignore[reportOperatorIssue]

    assert a_torch.grad is not None
    torch_grads = a_torch.grad.detach().numpy()

    assert jnp.allclose(korvax_grads, torch_grads, atol=1e-4, rtol=1e-1)


@pytest.mark.parametrize("order", [1, 2, 4, 32])
def test_allpole_outputs(x, order):
    a = jax.random.normal(jax.random.key(2), x.shape + (order,)) * 0.1

    y_korvax = jax.vmap(korvax.filter.ltv.allpole)(x, a)

    x_torch = torch.tensor(np.array(x), dtype=torch.float32)
    a_torch = torch.tensor(np.array(a), dtype=torch.float32, requires_grad=True)
    y_torch = torchlpc.sample_wise_lpc(x_torch, a_torch)

    assert jnp.allclose(y_korvax, y_torch.detach().numpy(), atol=1e-5)  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.parametrize("order", [1, 2, 4, 32])
def test_allpole_grads(x, order):
    a = jax.random.normal(jax.random.key(2), x.shape + (order,)) * 0.1

    x_torch = torch.tensor(np.array(x), dtype=torch.float32)
    a_torch = torch.tensor(np.array(a), dtype=torch.float32, requires_grad=True)
    y_torch = torchlpc.sample_wise_lpc(x_torch, a_torch)

    korvax_grads = jax.grad(
        lambda a: jnp.mean(jax.vmap(korvax.filter.ltv.allpole)(x, a=a) ** 2)
    )(a)

    (torch.mean(y_torch**2)).backward()  # pyright: ignore[reportOperatorIssue]

    assert a_torch.grad is not None
    torch_grads = a_torch.grad.detach().numpy()

    assert jnp.allclose(korvax_grads, torch_grads, atol=1e-4, rtol=1e-1)


@pytest.mark.parametrize("order", [1, 2, 4])
def test_fir(x, order):
    b = jax.random.normal(jax.random.key(3), x.shape + (order + 1,))

    y_out = jax.vmap(korvax.filter.ltv.fir)(x, b=b)

    y_ref = jax.vmap(korvax.filter.ltv.lfilter)(x, b=b)

    assert jnp.allclose(y_out, y_ref, atol=1e-5)
