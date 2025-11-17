# Transforms

The forward and inverse transforms follow the librosa convention: time-domain signals have the shape `(*batch, samples)`, and time-frequency representations have the shape shape `(*batch, bins, frames)`. 

## Forward Transforms

Functions that take in time-domain signals and output time-frequency representations.

::: korvax.stft

::: korvax.spectrogram

::: korvax.mel_spectrogram

::: korvax.mfcc

## Inverse Transforms

Functions that take in time-frequency representations and output time-domain signals.

::: korvax.istft

::: korvax.griffin_lim

## Utilities

Auxiliary functions used in various transforms.

::: korvax.mel_filterbank

::: korvax.cepstral_coefficients

::: korvax.to_mel_scale
