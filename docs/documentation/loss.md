# Loss Functions

If you're looking for an out-of-the-box MR-STFT loss, see [mrstft_loss](#korvax.loss.mrstft_loss).

Korvax provides a general interface for frame-based loss calculation. A loss is defined by three components: 

* a transform function that converts time-domain signals into time-frequency representations (e.g. STFT, VQT...)
* a loss function that computes a distance metric between two such representations (e.g. L1/L2, Wasserstein, spectral convergence...)
* an optional scaling function applied to each frame (e.g. Mel, A-weighting...)

This module contains documentation for the general interface, implements some common frame-wise loss functions, and a ready-to-use MR-STFT loss configuration. For transform functions implemented in Korvax, see [Transforms](transforms.md).

## General Interface

::: korvax.loss.time_frequency_loss
::: korvax.loss.TransformFn
::: korvax.loss.LossFn
::: korvax.loss.ScaleFn

## Loss Functions

::: korvax.loss.elementwise_loss
::: korvax.loss.spectral_convergence_loss
::: korvax.loss.spectral_optimal_transport_loss

## Ready-to-use Configurations

For convenience, common configurations of the above loss functions are provided.

::: korvax.loss.mrstft_loss
::: korvax.loss.smooth_mrstft_loss