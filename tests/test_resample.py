import pytest
import jax.numpy as jnp
import jax.random as jr
import torch
from torchaudio.functional import resample as rs_t

from korvax import resample as rs_j


@pytest.mark.parametrize(
    "seed,orig_sr,target_sr,lowpass_filter_width,rolloff",
    [
        (0, 48000, 44000, 6, 0.99),
        (1, 44100, 16000, 12, 0.95),
        (2, 32000, 48000, 8, 0.90),
    ],
)
def test_resample(seed, orig_sr, target_sr, lowpass_filter_width, rolloff):
    key = jr.key(seed)
    n = 4096 * 4
    x_jax = 0.5 * jr.normal(key=key, shape=(n,))
    x_jax += 0.5 * jnp.sin(
        2 * jnp.pi * 440 * jnp.linspace(0, n / orig_sr, num=n, endpoint=False)
    )

    x_torch = torch.tensor(x_jax)

    y_jax = rs_j(
        x_jax,
        orig_sr,
        target_sr,
        lowpass_filter_width=lowpass_filter_width,
        rolloff=rolloff,
    )
    y_torch = rs_t(
        x_torch[None, :],
        orig_sr,
        target_sr,
        lowpass_filter_width=lowpass_filter_width,
        rolloff=rolloff,
    ).squeeze(0)

    assert y_jax.shape == y_torch.shape
    assert jnp.allclose(y_jax, y_torch.numpy(), atol=1e-7)
