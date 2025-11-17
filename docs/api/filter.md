# Time-Domain Filtering

Time-domain filtering functions. The actual filtering is implemented in C++ kernels
and called via JAX's FFI interface. If a GPU is available, a CUDA kernel will be used.

This module is strongly based on [`philtorch`](https://github.com/yoyolicoris/philtorch/) and [`torchlpc`](https://github.com/DiffAPF/torchlpc/).

::: korvax.filter.allpole