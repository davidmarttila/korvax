# Time-Domain Filtering

Time-domain filtering functions that run efficiently on CPU and GPU, and are differentiable in all arguments. The implementations are strongly based on [`philtorch`](https://github.com/yoyolicoris/philtorch/) and [`torchlpc`](https://github.com/DiffAPF/torchlpc/).

## Time-Invariant Filtering

Functions that apply linear time-invariant (LTI) filters to time-domain signals. Filter coefficients are constant over time.

::: korvax.filter.lti.lfilter

::: korvax.filter.lti.sosfilt

## Time-Varying Filtering

Functions that apply linear time-varying (LTV) filters to time-domain signals. Filter coefficients can change at audio sample rate.

::: korvax.filter.ltv.lfilter

::: korvax.filter.ltv.sosfilt

::: korvax.filter.ltv.fir

::: korvax.filter.ltv.allpole