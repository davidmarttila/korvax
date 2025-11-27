# Time-Domain Filtering

Time-domain filtering functions. 

## Time-Invariant Filtering

Functions that apply linear time-invariant (LTI) filters to time-domain signals. Filter coefficients are constant over time.

::: korvax.filter.lfilter

::: korvax.filter.sosfilt

## Time-Varying Filtering

Functions that apply time-varying filters to time-domain signals. Filter coefficients can change at audio sample rate.

The actual filtering is implemented in C++ kernels
and called via JAX's FFI interface. If a GPU is available, a CUDA kernel will be used.

The implementations are strongly based on [`philtorch`](https://github.com/yoyolicoris/philtorch/) and [`torchlpc`](https://github.com/DiffAPF/torchlpc/).

::: korvax.filter.allpole
