import pytest
from pathlib import Path
import librosa
import scipy.io
import jax.numpy as jnp
import korvax
import numpy as np

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
