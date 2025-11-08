import pytest
import librosa
import numpy as np
import jax.numpy as jnp

from pathlib import Path
import korvax


@pytest.mark.parametrize("freq", [110, 220, 440, 880])
def test_yin_tone(freq):
    sr = 22050.0
    y = librosa.tone(freq, sr=sr, duration=1.0)
    f0 = korvax.pitch.yin(y, sr=sr, fmin=110.0, fmax=880.0, center=False)
    assert jnp.allclose(np.log2(f0), np.log2(freq), rtol=0, atol=1e-2)


def test_yin_chirp():
    # test yin on a chirp, using output from the vamp plugin as ground truth
    sr = 22050.0
    y = librosa.chirp(fmin=220, fmax=640, sr=sr, duration=1.0)
    f0 = korvax.pitch.yin(
        y,
        sr=sr,
        fmin=110.0,
        fmax=880.0,
        center=False,
        frame_length=1024,
        hop_length=512,
    )

    # adjust frames to the removal of win_length from yin
    f0 = f0[:-2]

    target_f0 = np.load(Path("tests") / "data" / "pitch-yin.npy")
    assert np.allclose(np.log2(f0), np.log2(target_f0), rtol=0, atol=1e-2)


def test_yin_chirp_instant():
    # test yin on a chirp, using frame-wise instantaneous frequency as ground truth
    sr = 22050.0
    chirp_min, chirp_max = 220, 640

    t = np.arange(sr) / sr
    f = chirp_min * (chirp_max / chirp_min) ** t

    fl = 2048
    hl = 512

    y = librosa.chirp(fmin=chirp_min, fmax=chirp_max, sr=sr, duration=1.0, linear=False)
    target_f0 = librosa.util.frame(f, frame_length=fl, hop_length=hl).mean(axis=0)

    f0 = korvax.pitch.yin(
        y, fmin=110.0, fmax=880.0, sr=sr, frame_length=fl, hop_length=hl, center=False
    )
    assert jnp.allclose(jnp.log2(f0), jnp.log2(target_f0), rtol=0, atol=1e-2)


@pytest.mark.parametrize("freq", [110.0, 220.0, 440.0, 880.0])
def test_pyin_tone(freq):
    sr = 22050.0
    y = librosa.tone(freq, sr=sr, duration=1.0)
    f0, _, _ = korvax.pitch.pyin(y, sr=sr, fmin=110.0, fmax=1000.0, center=False)
    assert jnp.allclose(jnp.log2(f0), jnp.log2(freq), rtol=0, atol=1e-2)
