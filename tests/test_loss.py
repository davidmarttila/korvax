import pytest
import jax
import numpy as np
import jax.numpy as jnp
import torch

import korvax

from auraloss.freq import (
    MultiResolutionSTFTLoss,
    STFTMagnitudeLoss,
    MelSTFTLoss,
    SpectralConvergenceLoss,
)


@pytest.fixture
def signals():
    key = jax.random.key(0)
    k1, k2 = jax.random.split(key)
    with jax.enable_x64():
        x = jax.random.normal(k1, (4, 7, 16000))
        y = jax.random.normal(k2, (4, 7, 16000))
    return x, y


@pytest.fixture
def spectra(signals):
    x, y = signals
    res = 1024
    with jax.enable_x64():
        X = jnp.abs(korvax.stft(x, res, res // 4, res // 2))
        Y = jnp.abs(korvax.stft(y, res, res // 4, res // 2))
    return X, Y


@pytest.mark.parametrize("log", [True, False])
@pytest.mark.parametrize("reduction", ["sum", "mean"])
@pytest.mark.parametrize("mag_distance", ["L1", "L2"])
def test_stft_magnitude_loss(spectra, log, reduction, mag_distance):
    with jax.enable_x64():
        X, Y = spectra
        loss = korvax.loss.stft_magnitude_loss(
            X, Y, log=log, reduction=reduction, distance=mag_distance
        )
        loss_ref = (
            STFTMagnitudeLoss(log, mag_distance, reduction)(
                torch.from_numpy(np.array(X)),
                torch.from_numpy(np.array(Y)),
            )
            .detach()
            .numpy()
        )
        assert jnp.allclose(loss, loss_ref)


def test_spectral_convergence_loss(spectra):
    with jax.enable_x64():
        X, Y = spectra
        loss = korvax.loss.spectral_convergence_loss(X, Y)
        loss_ref = (
            SpectralConvergenceLoss()(
                torch.from_numpy(np.array(X)),
                torch.from_numpy(np.array(Y)),
            )
            .detach()
            .numpy()
        )
        assert jnp.allclose(loss, loss_ref)


def test_multi_resolution_stft_loss(signals):
    with jax.enable_x64():
        x, y = signals
        loss, (sc_losses, log_mag_losses, lin_mag_losses, phs_losses) = (
            korvax.loss.multi_resolution_stft_loss(
                x,
                y,
                fft_sizes=[1024, 2048, 512],
                hop_sizes=[120, 240, 50],
                win_lengths=[600, 1200, 240],
                output="full",
                w_lin_mag=1.0,
                w_log_mag=1.0,
                w_sc=1.0,
                w_phs=1.0,
                log_eps=0.0,
            )
        )
        mrstft_loss = MultiResolutionSTFTLoss(
            fft_sizes=[1024, 2048, 512],
            hop_sizes=[120, 240, 50],
            win_lengths=[600, 1200, 240],
            output="full",
            w_lin_mag=1.0,
            w_log_mag=1.0,
            w_sc=1.0,
            w_phs=1.0,
        )
        (
            loss_ref,
            sc_losses_ref,
            log_mag_losses_ref,
            lin_mag_losses_ref,
            phs_losses_ref,
        ) = mrstft_loss(
            torch.from_numpy(np.array(x)),
            torch.from_numpy(np.array(y)),
        )
        loss_ref = loss_ref.detach().numpy()
        sc_losses_ref = np.concatenate(
            [ls.unsqueeze(0) for ls in sc_losses_ref], axis=0
        )
        log_mag_losses_ref = np.concatenate(
            [ls.unsqueeze(0) for ls in log_mag_losses_ref], axis=0
        )
        lin_mag_losses_ref = np.concatenate(
            [ls.unsqueeze(0) for ls in lin_mag_losses_ref], axis=0
        )
        phs_losses_ref = np.concatenate(
            [ls.unsqueeze(0) for ls in phs_losses_ref], axis=0
        )

        assert jnp.allclose(loss, loss_ref, atol=1e-1, rtol=1e-1)
        assert jnp.allclose(
            jnp.array(sc_losses), jnp.array(sc_losses_ref), atol=1e-1, rtol=1e-1
        )
        assert jnp.allclose(
            jnp.array(log_mag_losses),
            jnp.array(log_mag_losses_ref),
            atol=1e-1,
            rtol=1e-1,
        )
        assert jnp.allclose(
            jnp.array(lin_mag_losses),
            jnp.array(lin_mag_losses_ref),
            atol=1e-1,
            rtol=1e-1,
        )
        assert jnp.allclose(
            jnp.array(phs_losses), jnp.array(phs_losses_ref), atol=1e-1, rtol=1e-1
        )


def test_mel_stft_loss(signals):
    with jax.enable_x64():
        x, y = signals
        x = x.astype(jnp.float32)
        y = y.astype(jnp.float32)
    loss = korvax.loss.stft_loss(
        x, y, filterbank=korvax.mel_filterbank(sr=16000.0, n_fft=1024, n_mels=80)
    )
    loss_ref = (
        MelSTFTLoss(
            n_mels=80,
            fft_size=1024,
            hop_size=256,
            win_length=1024,
            sample_rate=16000,
            reduction="mean",
        )(
            torch.from_numpy(np.array(x)),
            torch.from_numpy(np.array(y)),
        )
        .detach()
        .numpy()
    )
    assert jnp.allclose(loss, loss_ref, atol=1e-2, rtol=1e-1)
