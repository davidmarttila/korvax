from functools import partial
import torch
import pytest
from pathlib import Path
import librosa
import scipy.io
import jax.numpy as jnp
import korvax
import numpy as np
import jax

import korvax.transforms._cqt as kcqt
import nnAudio
import nnAudio.utils
import nnAudio.features

data_dir = Path("tests/data/")


@pytest.fixture
def x():
    return jax.random.normal(jax.random.key(0), (5, 16000))


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


@pytest.fixture
def y_chirp():
    sr = 22050
    y = librosa.chirp(fmin=55, fmax=55 * 2**7, length=sr // 8, sr=sr)
    return y


@pytest.mark.parametrize("hop_length", [None, 1024])
@pytest.mark.parametrize("win_length", [None, 1024])
@pytest.mark.parametrize("n_fft", [2048, 2049])
@pytest.mark.parametrize("window", ["hann", "boxcar"])
@pytest.mark.parametrize("center", [False, True])
@pytest.mark.parametrize("use_length", [False, True])
@pytest.mark.parametrize("pad_mode", ["constant", "reflect"])
@pytest.mark.parametrize("momentum", [0.0, 0.99])
def test_griffinlim(
    y_chirp,
    hop_length,
    win_length,
    n_fft,
    window,
    center,
    use_length,
    pad_mode,
    momentum,
):
    if use_length:
        length = len(y_chirp)
    else:
        length = None

    pad_kwargs = {"mode": pad_mode}

    D = korvax.stft(
        y_chirp,
        hop_length=hop_length,
        win_length=win_length,
        n_fft=n_fft,
        window=window,
        center=center,
        pad_kwargs=pad_kwargs,
    )

    S = jnp.abs(D)

    y_rec = korvax.griffin_lim(
        S,
        hop_length=hop_length,
        win_length=win_length,
        n_fft=n_fft,
        window=window,
        center=center,
        length=length,
        pad_kwargs=pad_kwargs,
        n_iter=3,
        momentum=momentum,
    )

    if use_length:
        assert len(y_rec) == length


@pytest.fixture(scope="module")
def y_22050():
    y, sr = librosa.load(Path("tests") / "data" / "test1_22050.wav")
    return y


@pytest.mark.parametrize("n_fft", [1024, 755, 2048, 2049])
@pytest.mark.parametrize("hop_length", [None, 512])
@pytest.mark.parametrize("power", [1.0, 2.0])
def test_spectrogram(y_22050, n_fft, hop_length, power):
    y = y_22050
    S = jnp.abs(korvax.stft(y, n_fft=n_fft, hop_length=hop_length)) ** power

    S_ = korvax.spectrogram(y, n_fft=n_fft, hop_length=hop_length, power=power)

    assert jnp.allclose(S, S_)

    S_ = korvax.spectrogram(y, n_fft=n_fft, hop_length=hop_length, power=power)
    assert jnp.allclose(S, S_)


@pytest.mark.parametrize("infile", files("feature-melfb-*.mat"))
@pytest.mark.filterwarnings("ignore:Empty filters detected")
def test_melfb(infile):
    DATA = load(infile)

    with jax.enable_x64():
        wts = korvax.mel_filterbank(
            sr=float(DATA["sr"][0, 0]),
            n_fft=int(DATA["nfft"][0, 0]),
            n_mels=int(DATA["nfilts"][0, 0]),
            fmin=float(DATA["fmin"][0, 0]),
            fmax=float(DATA["fmax"][0, 0]),
            htk=bool(DATA["htk"][0, 0]),
            dtype=DATA["wts"].dtype,
        )

    wts = jnp.pad(wts, [(0, 0), (0, DATA["nfft"][0, 0] // 2 - 1)], mode="constant")

    assert wts.shape == DATA["wts"].shape
    assert np.allclose(wts, DATA["wts"])


@pytest.mark.parametrize(
    "S", [librosa.power_to_db(np.random.randn(128, 1) ** 2, ref=np.max)]
)
@pytest.mark.parametrize("norm", [None, "ortho"])
@pytest.mark.parametrize("n_mfcc", [13, 20])
@pytest.mark.parametrize("lifter", [0.0, 13.0])
def test_mfcc(S, norm, n_mfcc, lifter):
    E_total = np.sum(S, axis=0)

    with jax.enable_x64():
        mfcc_lib = librosa.feature.mfcc(
            S=S, dct_type=2, norm=norm, n_mfcc=n_mfcc, lifter=lifter
        )

        mfcc_kvx = korvax.cepstral_coefficients(
            jnp.asarray(S),
            norm=norm,
            n_cc=n_mfcc,
            lifter=lifter,
            mag_scale="linear",
        )

        # In type-2 mode, DC component should be constant over all frames
        assert jnp.var(mfcc_kvx[0] / E_total) <= 1e-29

        assert jnp.allclose(mfcc_kvx, mfcc_lib)


@pytest.mark.parametrize("Q", [0.1, 1.0, 2.0])
@pytest.mark.parametrize("n_bins", [12, 36])
@pytest.mark.parametrize("bins_per_octave", [12, 24])
@pytest.mark.parametrize("fmax", [None, 8000.0])
@pytest.mark.parametrize("gamma", [0.0, 1.0])
def test_create_vqt_kernels(Q, n_bins, bins_per_octave, fmax, gamma):
    sr = 16000.0
    fmin = 300.0
    norm = 1
    window = "hann"
    topbin_check = True

    kernels, lengths, freqs = kcqt.create_vqt_kernels(
        Q=Q,
        sr=sr,
        fmin=fmin,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        norm=norm,
        window=window,
        fmax=fmax,
        topbin_check=topbin_check,
        gamma=gamma,
        dtype=jnp.float32,
    )

    ref_kernels, _, ref_lengths, ref_freqs = nnAudio.utils.create_cqt_kernels(
        Q=Q,
        fs=sr,
        fmin=fmin,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        norm=norm,
        window=window,
        fmax=fmax,
        gamma=gamma,  # pyright: ignore[reportArgumentType]
    )

    assert jnp.allclose(kernels, jnp.asarray(ref_kernels), atol=1e-5)


@pytest.mark.parametrize("filter_scale", [0.1, 1.0, 2.0])
def test_vqt(x, filter_scale):
    sr = 16000.0
    fmin = 300.0
    norm = 1
    window = "hann"

    out = jax.vmap(
        partial(
            kcqt.vqt,
            sr=sr,
            filter_scale=filter_scale,
            fmin=fmin,
            n_bins=24,
            norm_kernels=norm,
            window=window,
            power=1.0,
        )
    )(x)

    t_cqt = nnAudio.features.CQT1992v2(
        sr=int(sr),
        fmin=fmin,
        n_bins=24,
        bins_per_octave=12,
        filter_scale=filter_scale,
        norm=norm,
        window=window,
        output_format="Magnitude",
        verbose=False,
        pad_mode="constant",
    )

    ref_out = t_cqt(torch.tensor(np.asarray(x)))

    assert out.shape == jnp.asarray(ref_out).shape
    assert jnp.allclose(out, jnp.asarray(ref_out), atol=1e-4)
