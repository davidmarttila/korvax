# Home

Korvax ("korva" means "ear" in Finnish) is a Python package for audio signal processing and machine learning in JAX. Korvax is framework-agnostic; it can be used in Flax, Equinox, or any other JAX framework.

JAX is a very powerful library for signal processing related tasks because of its composable function transformations (e.g., `jit`, `vmap`, `grad`, `pmap`, etc.), and often faster than PyTorch. However, PyTorch has a lot more available packages for audio processing. Korvax aims to fill this gap for JAX by porting functionality from packages such as:

* [torchaudio](https://pytorch.org/audio/stable/index.html)
* [librosa](https://librosa.org/doc/latest/index.html)
* [philtorch](https://github.com/yoyolicoris/philtorch)
* [nnAudio](https://github.com/KinWaiCheuk/nnAudio)
* [sot-loss](https://github.com/kan-bayashi/sot-loss)

## Installation

Install it via pip:

```bash
pip install korvax
```

Korvax provides a `[cuda12]` option that simply makes sure that `jax[cuda12]` is installed:

```bash
pip install "korvax[cuda12]"
```

## Features

Currently, Korvax provides:

- Time-frequency loss functions (MR-STFT, Spectral Optimal Transport)
- Pitch estimation algorithms
- Time-varying all-pole filtering
- Resampling
- Transforms (STFT, Mel spectrogram, MFCCs, etc.)

All features are GPU-ready and differentiable.

See the documentation in the sidebar for more details and usage examples.