#include "xla/ffi/api/ffi.h"
#include "nanobind/nanobind.h"
#include "_common.h"

namespace nb = nanobind;
namespace ffi = xla::ffi;

#define XLA_CHECK_ARG(cond, message) \
    if (!(cond))                     \
        return ffi::Error::InvalidArgument(message);

template <typename scalar_t>
inline void samplewise_allpole(int T, int order,
                               const scalar_t *a,
                               scalar_t *out)
{
    auto out_offset = out + order;
    for (int64_t t = 0; t < T; t++)
    {
        scalar_t y = out_offset[t];
        for (int64_t i = 0; i < order; i++)
        {
            y -= a[t * order + i] * out_offset[t - i - 1];
        }
        out_offset[t] = y;
    }
}

template <typename scalar_t>
void batched_samplewise_allpole(int B, int T, int order,
                                const scalar_t *a,
                                scalar_t *out)
{
#pragma omp parallel for
    for (auto b = 0; b < B; b++)
        samplewise_allpole<scalar_t>(T, order, a + b * T * order, out + b * (T + order));
}

ffi::Error allpole_impl(ffi::Buffer<ffi::F32> x,
                        ffi::Buffer<ffi::F32> a,
                        ffi::ResultBuffer<ffi::F32> out)
{
    int x_ndim = x.dimensions().size();
    int a_ndim = a.dimensions().size();

    int x_len, order;

    if (x_ndim == 1)
    {
        x_len = static_cast<int>(x.dimensions()[0]);
        std::copy_n(x.typed_data(), x_len, out->typed_data());

        XLA_CHECK_ARG(a_ndim == 2 && a.dimensions()[0] == x_len - static_cast<int>(a.dimensions()[1]),
                      "Input buffer `a` must have shape [T, order] when input buffer `x` has shape [T + order].");

        order = static_cast<int>(a.dimensions()[0]);
        samplewise_allpole<float>(x_len - order, order, a.typed_data(), out->typed_data());
        return ffi::Error::Success();
    }

    XLA_CHECK_ARG(x_ndim == 2,
                  "Input buffer `x` must have shape [T + order] or [B, T + order].");

    int B = static_cast<int>(x.dimensions()[0]);

    x_len = static_cast<int>(x.dimensions()[1]);

    std::copy_n(x.typed_data(), B * x_len, out->typed_data());

    XLA_CHECK_ARG(a_ndim == 3 && a.dimensions()[0] == B && a.dimensions()[1] == x_len - static_cast<int>(a.dimensions()[2]),
                  "Input buffer `a` must have shape [B, T, order] when input buffer `x` has shape [B, T + order].");
    order = static_cast<int>(a.dimensions()[2]);
    batched_samplewise_allpole<float>(B, x_len - order, order, a.typed_data(),
                                      out->typed_data());
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    allpole, allpole_impl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::F32>>() // x
        .Arg<ffi::Buffer<ffi::F32>>() // a
        .Ret<ffi::Buffer<ffi::F32>>() // out
);

NB_MODULE(_filter_cpu, m)
{
    m.def("allpole", []()
          { return EncapsulateFfiCall(allpole); });
}
