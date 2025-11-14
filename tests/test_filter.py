import pytest
import jax
import jax.test_util
import jax.numpy as jnp
import numpy as np

import torchlpc
import torch

# from torchaudio.functional import lfilter as torch_lfilter
import korvax

# from korvax.filter import lfilter as korvax_lfilter
# from scipy.signal import lfilter as scipy_lfilter


@pytest.fixture
def x():
    return jax.random.normal(jax.random.key(0), (5, 1000))


@pytest.fixture
def a():
    return jnp.array([1.0, -0.95, 0.7, -0.3])


@pytest.fixture
def b():
    return jnp.array([0.3, 0.1, 0.2, 0.4])


# def test_lfilter_output(b, a, x):
#     y_korvax = jax.vmap(korvax_lfilter, in_axes=(0, None, None))(x, a, b)
#     y_scipy = scipy_lfilter(
#         np.array(b, dtype=np.float32),
#         np.array(a, dtype=np.float32),
#         np.array(x, dtype=np.float32),
#     )
#     y_torch = torch_lfilter(
#         torch.tensor(x, dtype=torch.float32),
#         torch.tensor(a, dtype=torch.float32),
#         torch.tensor(b, dtype=torch.float32),
#         clamp=False,
#     ).numpy()

#     assert jnp.allclose(y_korvax, jnp.array(y_scipy), atol=1e-5)
#     assert jnp.allclose(y_korvax, y_torch, atol=1e-5)


# def test_lfilter_grads(b, a, x):
#     korvax_grad_fn = jax.grad(
#         lambda b, a, x: jnp.mean(
#             jax.vmap(korvax_lfilter, in_axes=(0, None, None))(x, a, b) ** 2
#         ),
#         argnums=(0, 1),
#     )

#     x_torch = torch.tensor(x, dtype=torch.float32)
#     a_torch = torch.tensor(a, dtype=torch.float32, requires_grad=True)
#     b_torch = torch.tensor(b, dtype=torch.float32, requires_grad=True)

#     def torch_loss_fn(b, a, x):
#         y = torch_lfilter(x, a, b, clamp=False)
#         return torch.mean(y**2)

#     korvax_grads = korvax_grad_fn(b, a, x)
#     torch_loss = torch_loss_fn(b_torch, a_torch, x_torch)
#     torch_loss.backward()
#     torch_grads = np.asarray(b_torch.grad), np.asarray(a_torch.grad)

#     assert jnp.allclose(korvax_grads[0], jnp.asarray(torch_grads[0]), atol=1e-5)
#     assert jnp.allclose(korvax_grads[1], jnp.asarray(torch_grads[1]), atol=1e-5)


@pytest.mark.parametrize("order", [1, 2, 4, 6])
def test_allpole_values(x, order):
    a = jax.random.normal(jax.random.key(1), x.shape + (order,)) * 0.1

    y_korvax = jax.vmap(korvax.filter.allpole)(x, a)

    x_torch = torch.tensor(np.array(x), dtype=torch.float32)
    a_torch = torch.tensor(np.array(a), dtype=torch.float32, requires_grad=True)
    y_torch = torchlpc.sample_wise_lpc(x_torch, a_torch)

    assert jnp.allclose(y_korvax, y_torch.detach().numpy(), atol=1e-5)


@pytest.mark.parametrize("order", [1, 2, 4, 6])
def test_allpole_grads(x, order):
    a = jax.random.normal(jax.random.key(1), x.shape + (order,)) * 0.1

    x_torch = torch.tensor(np.array(x), dtype=torch.float32)
    a_torch = torch.tensor(np.array(a), dtype=torch.float32, requires_grad=True)
    y_torch = torchlpc.sample_wise_lpc(x_torch, a_torch)

    korvax_grads = jax.grad(
        lambda a: jnp.mean(jax.vmap(korvax.filter.allpole)(x, a) ** 2)
    )(a)

    (torch.mean(y_torch**2)).backward()

    assert a_torch.grad is not None
    torch_grads = a_torch.grad.detach().numpy()

    assert jnp.allclose(korvax_grads, torch_grads, atol=1e-5)
