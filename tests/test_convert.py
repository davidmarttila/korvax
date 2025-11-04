import pytest
import jax
import jax.numpy as jnp
import numpy as np
import librosa


import korvax


def test_midi_to_hz():
    notes = np.array([0, 21, 60, 69, 84, 127.0])
    expected = librosa.midi_to_hz(notes)
    got = korvax.midi_to_hz(notes)
    assert jnp.allclose(got, expected)


def test_hz_to_midi():
    freqs = np.array([8.1757989156, 55.0, 440.0, 4186.009044809578])
    expected = librosa.hz_to_midi(freqs)
    got = korvax.hz_to_midi(freqs)
    assert jnp.allclose(got, expected)


@pytest.mark.parametrize("htk", [False, True])
def test_hz_to_mel(htk):
    freqs = np.linspace(0.0, 11025.0, num=7)[1:]
    expected = librosa.hz_to_mel(freqs, htk=htk)
    got = korvax.hz_to_mel(freqs, htk=htk)
    assert jnp.allclose(got, expected)


@pytest.mark.parametrize("htk", [False, True])
def test_mel_to_hz(htk):
    mels = np.linspace(0.0, 1000.0, num=7)[1:]
    expected = librosa.mel_to_hz(mels, htk=htk)
    got = korvax.mel_to_hz(mels, htk=htk)
    assert jnp.allclose(got, expected)


def test_fft_frequencies():
    expected = librosa.fft_frequencies(sr=22050, n_fft=2048)
    got = korvax.fft_frequencies(sr=22050.0, n_fft=2048)
    assert jnp.allclose(got, expected)


def test_mel_frequencies():
    expected = librosa.mel_frequencies(n_mels=10, fmin=0.0, fmax=11025.0, htk=False)
    got = korvax.mel_frequencies(10, fmin=0.0, fmax=11025.0, htk=False)
    assert jnp.allclose(got, expected)


def test_cqt_frequencies():
    expected = librosa.cqt_frequencies(12, fmin=16.35, bins_per_octave=12, tuning=0.0)
    got = korvax.cqt_frequencies(12, fmin=16.35, bins_per_octave=12, tuning=0.0)
    assert jnp.allclose(got, expected)


def test_A_weighting():
    with jax.enable_x64():
        freqs = np.logspace(np.log10(20.0), np.log10(20000.0), num=50)
        expected = librosa.A_weighting(freqs, min_db=None)
        got = korvax.A_weighting(freqs, min_db=None)
    assert jnp.allclose(got, expected)


def test_B_weighting():
    with jax.enable_x64():
        freqs = np.logspace(np.log10(20.0), np.log10(20000.0), num=50)
        expected = librosa.B_weighting(freqs, min_db=None)
        got = korvax.B_weighting(freqs, min_db=None)
    assert jnp.allclose(got, expected)


def test_C_weighting():
    with jax.enable_x64():
        freqs = np.logspace(np.log10(20.0), np.log10(20000.0), num=50)
        expected = librosa.C_weighting(freqs, min_db=None)
        got = korvax.C_weighting(freqs, min_db=None)
    assert jnp.allclose(got, expected)


def test_D_weighting():
    with jax.enable_x64():
        freqs = np.logspace(np.log10(20.0), np.log10(20000.0), num=50)
        expected = librosa.D_weighting(freqs, min_db=None)
        got = korvax.D_weighting(freqs, min_db=None)
    assert jnp.allclose(got, expected)


def test_power_to_db():
    S = np.abs(
        librosa.stft(librosa.chirp(fmin=32, fmax=8192, sr=22050, duration=2.0))
    ).astype(np.float32)
    expected = librosa.power_to_db(S**2, ref=np.max, amin=1e-8, top_db=80.0)
    got = korvax.power_to_db(S**2, ref=jnp.max, amin=1e-8, top_db=80.0)
    assert jnp.allclose(got, expected, atol=1e-5)


def test_db_to_power():
    S_db = np.array([-80.0, -40.0, -20.0, 0.0, 20.0, 40.0], dtype=np.float32)
    expected = librosa.db_to_power(S_db, ref=2.0)
    got = korvax.db_to_power(S_db, ref=2.0)
    assert jnp.allclose(got, expected, atol=1e-5)


def test_amplitude_to_db():
    S = np.abs(
        librosa.stft(librosa.chirp(fmin=32, fmax=8192, sr=22050, duration=2.0))
    ).astype(np.float32)
    expected = librosa.amplitude_to_db(S, ref=np.max, amin=1e-8, top_db=80.0)
    got = korvax.amplitude_to_db(S, ref=jnp.max, amin=1e-8, top_db=80.0)
    assert jnp.allclose(got, expected, atol=1e-5)


def test_db_to_amplitude():
    S_db = np.array([-80.0, -40.0, -20.0, 0.0, 20.0, 40.0], dtype=np.float32)
    expected = librosa.db_to_amplitude(S_db, ref=2.0)
    got = korvax.db_to_amplitude(S_db, ref=2.0)
    assert jnp.allclose(got, expected, atol=1e-5)
