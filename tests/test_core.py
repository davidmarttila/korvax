import pytest
from pathlib import Path
import librosa
import scipy.io
import jax.numpy as jnp
import korvax
import numpy as np
import jax

data_dir = Path("tests/data/")


def files(pattern: str) -> list[Path]:
    out = list(data_dir.glob(pattern))
    return sorted(out)


def load(infile):
    return scipy.io.loadmat(str(infile), chars_as_strings=True)


@pytest.mark.parametrize("infile", files("core-stft-*.mat"))
def test_stft(infile):
    DATA = load(infile)

    x, _ = librosa.load(Path("tests") / Path(DATA["wavfile"][0]), sr=None, mono=True)

    if DATA["hann_w"][0, 0] == 0:
        window = "boxcar"
        win_length = None

    else:
        window = "hann"
        win_length = int(DATA["hann_w"][0, 0])

    k_stft = korvax.stft(
        jnp.asarray(x),
        n_fft=int(DATA["nfft"][0, 0]),
        hop_length=int(DATA["hop_length"][0, 0]),
        win_length=win_length,
        window=window,
        center=False,
    )

    l_stft = jnp.asarray(
        librosa.stft(
            x.astype(np.float32),
            n_fft=int(DATA["nfft"][0, 0]),
            hop_length=int(DATA["hop_length"][0, 0]),
            win_length=win_length,
            window=window,
            center=False,
            dtype=np.complex64,
        )
    )

    assert jnp.allclose(k_stft.real, l_stft.real, atol=1e-5)
    assert jnp.allclose(k_stft.imag, l_stft.imag, atol=1e-5)


@pytest.fixture(scope="module", params=[22050, 44100])
def y_chirp_istft(request):
    sr = request.param
    return (librosa.chirp(fmin=32, fmax=8192, sr=sr, duration=2.0), sr)


@pytest.mark.parametrize("n_fft", [1024, 1025, 2048, 4096])
@pytest.mark.parametrize("window", ["hann", "blackmanharris"])
@pytest.mark.parametrize("hop_length", [128, 256, 512])
@pytest.mark.parametrize("center", [True, False])
def test_istft_reconstruction(y_chirp_istft, n_fft, hop_length, window, center):
    with jax.enable_x64():
        x, sr = y_chirp_istft
        x = jnp.asarray(x, dtype=jnp.float64)
        S = korvax.stft(
            x, n_fft=n_fft, hop_length=hop_length, window=window, center=center
        )
        x_reconstructed = korvax.istft(
            S,
            hop_length=hop_length,
            window=window,
            n_fft=n_fft,
            length=len(x) if center else None,
            center=center,
        )

        if not center:
            x = korvax.util.fix_length(x, x_reconstructed.shape[-1])

        # NaN/Inf/-Inf should not happen
        assert jnp.all(jnp.isfinite(x_reconstructed))

        # should be almost approximately reconstructed
        assert jnp.allclose(x, x_reconstructed, atol=1e-6)
