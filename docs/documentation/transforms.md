# Transforms

The forward and inverse transforms follow the librosa convention: time-domain signals have the shape `(*batch, samples)`, and time-frequency representations have the shape `(*batch, bins, frames)`. 

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

## Frequency Transforms

These functions take in frequency-domain representations and output modified frequency-domain representations. They are used in the above time-to-frequency transforms, but can also be used standalone.

::: korvax.cepstral_coefficients

::: korvax.to_mel_scale

## Perceptual Loudness Weighting

::: korvax.A_weighting

::: korvax.B_weighting

::: korvax.C_weighting

::: korvax.D_weighting

## Utilities

::: korvax.mel_filterbank

::: korvax.mel_to_hz

::: korvax.hz_to_mel

::: korvax.db_to_amplitude

::: korvax.amplitude_to_db

::: korvax.power_to_db

::: korvax.db_to_power

::: korvax.fft_frequencies

::: korvax.mel_frequencies

::: korvax.cqt_frequencies