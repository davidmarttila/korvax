import pytest
import jax
import jax.numpy as jnp
import numpy as np

import torch
from torchaudio.functional import lfilter as torch_lfilter

from korvax.filter import lfilter as korvax_lfilter
from scipy.signal import lfilter as scipy_lfilter


@pytest.fixture
def x():
    return jax.random.normal(jax.random.key(0), (5, 3, 1000))


def test_lfilter_output(x):
    b = jnp.array([0.3, 0.1, 0.2, 0.4])
    a = jnp.array([1.0, -0.95, 0.7, -0.3])

    y_korvax = korvax_lfilter(b, a, x)
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


def main():
    t_in = torch.randn(2, 16000)
    a_coeffs = torch.tensor([1.0, -0.95, 0.7, -0.3])
    b_coeffs = torch.tensor([0.3, 0.1, 0.2, 0.4])
    print(torch_lfilter(t_in, a_coeffs, b_coeffs).shape)


if __name__ == "__main__":
    main()
