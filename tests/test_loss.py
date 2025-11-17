import pytest
import jax
import numpy as np
import jax.numpy as jnp
import torch

import korvax
from functools import partial

from auraloss.freq import (
    MultiResolutionSTFTLoss,
    MelSTFTLoss,
)

from sot import Wasserstein1DLoss


@pytest.fixture
def signals():
    key = jax.random.key(0)
    k1, k2 = jax.random.split(key)
    x = jax.random.normal(k1, (4, 7, 16000))
    y = jax.random.normal(k2, (4, 7, 16000))
    return x, y


def test_multi_resolution_stft_loss(signals):
    x, y = signals
    loss = korvax.loss.mrstft_loss(
        x,
        y,
        fft_sizes=[1024, 2048, 512],
        hop_lengths=[120, 240, 50],
        win_lengths=[600, 1200, 240],
        w_lin=1.0,
        w_log=1.0,
        log_eps=0.0,
    )
    mrstft_loss = MultiResolutionSTFTLoss(
        fft_sizes=[1024, 2048, 512],
        hop_sizes=[120, 240, 50],
        win_lengths=[600, 1200, 240],
        w_lin_mag=1.0,
        w_log_mag=1.0,
        w_sc=0.0,
        w_phs=0.0,
    )
    loss_ref = mrstft_loss(
        torch.from_numpy(np.array(x)),
        torch.from_numpy(np.array(y)),
    )
    loss_ref = loss_ref.detach().numpy()

    assert jnp.allclose(loss, loss_ref, atol=1e-2, rtol=1e-2)


def test_mel_stft_loss(signals):
    x, y = signals

    def mel_loss(x, y):
        return korvax.loss.time_frequency_loss(
            x,
            y,
            transform_fn=partial(
                korvax.mel_spectrogram,
                sr=16000.0,
                n_fft=1024,
                hop_length=256,
                win_length=1024,
                n_mels=80,
                power=1,
            ),
            loss_fn=partial(korvax.loss.elementwise_loss, metric="L1"),
        )

    loss = mel_loss(x, y)
    loss_ref = (
        MelSTFTLoss(
            n_mels=80,
            fft_size=1024,
            hop_size=256,
            win_length=1024,
            sample_rate=16000,
            w_log_mag=0.0,
            w_lin_mag=1.0,
            w_sc=0.0,
            reduction="mean",
        )(
            torch.from_numpy(np.array(x)),
            torch.from_numpy(np.array(y)),
        )
        .detach()
        .numpy()
    )
    assert jnp.allclose(loss, loss_ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("balanced", [True, False])
@pytest.mark.parametrize("quantile_lowpass", [True, False])
@pytest.mark.parametrize("p", [1, 2])
def test_stft_sot_loss(signals, normalize, balanced, quantile_lowpass, p):
    x, y = signals
    loss = korvax.loss.time_frequency_loss(
        x,
        y,
        transform_fn=partial(
            korvax.spectrogram,
            hop_length=512,
            win_length=2048,
            window="flattop",
            power=2,
        ),
        loss_fn=partial(
            korvax.loss.spectral_optimal_transport_loss,
            balanced=balanced,
            normalize=normalize,
            quantile_lowpass=quantile_lowpass,
            p=p,
        ),
    )
    loss_ref = (
        Wasserstein1DLoss(
            transform="stft",
            hop_length=512,
            fft_size=2048,
            window="flattop",
            square_magnitude=True,
            sample_rate=16000,
            balanced=balanced,
            normalize=normalize,
            quantile_lowpass=quantile_lowpass,
            p=p,
        )(
            torch.from_numpy(np.array(x)),
            torch.from_numpy(np.array(y)),
        )
        .detach()
        .numpy()
    )

    assert jnp.allclose(loss, loss_ref, atol=1e-2, rtol=1e-1)
