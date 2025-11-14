#include "xla/ffi/api/ffi.h"
#include "_common.h"

namespace ffi = xla::ffi;
template <typename scalar_t>
__global__ void allpole_kernel(scalar_t *out,     // [B, T + order]
                               const scalar_t *A, // [B, T, order]
                               int64_t B, int64_t T, int64_t order)
{
    extern __shared__ char smem[];
    scalar_t *sm = reinterpret_cast<scalar_t *>(smem);

    int b = blockIdx.x;
    int i = threadIdx.x;

    if (b >= B || i >= order)
        return;

    // Initialize shared memory with the first 'order' elements
    sm[i] = padded_y[b * (T + order) + i];
    __syncthreads();

    int circular_idx = 0;
    for (int t = 0; t < T; ++t)
    {
        circular_idx = t % order;
        scalar_t a = -A[((b * T + t) * order) + i];

        // Compute s as in the Python code
        int idx_offset = circular_idx - i - 1;
        if (i > circular_idx - 1)
        {
            idx_offset += order;
        }
        scalar_t s = sm[(idx_offset + order) % order];

        scalar_t v = a * s;

        if (i == order - 1)
        {
            sm[circular_idx] = v;
            v = padded_y[b * (T + order) + t + order];
        }
        __syncthreads();

        // Atomic add to shared memory
        atomicAdd(&sm[circular_idx], v);
        __syncthreads();

        if (i == order - 1)
        {
            padded_y[b * (T + order) + t + order] = sm[circular_idx];
        }
        __syncthreads();
    }
}

ffi::Error allpole_impl(cudaStream_t stream,
                        ffi::Buffer<ffi::F32> x,
                        ffi::Buffer<ffi::F32> a,
                        ffi::ResultBuffer<ffi::F32> out)
{
    int x_ndim = x.dimensions().size();
    int a_ndim = a.dimensions().size();

    int B, x_len, order;

    if (x_ndim == 1)
    {
        XLA_CHECK_ARG(a_ndim == 2 && a.dimensions()[0] == x.dimensions()[0] - static_cast<int>(a.dimensions()[1]),
                      "Input buffer `a` must have shape [T, order] when input buffer `x` has shape [T + order].");

        B = 1;
        x_len = static_cast<int>(x.dimensions()[0]);
        order = static_cast<int>(a.dimensions()[1]);
    }
    else
    {
        XLA_CHECK_ARG(x_ndim == 2,
                      "Input buffer `x` must have shape [T + order] or [B, T + order].");

        XLA_CHECK_ARG(a_ndim == 3 && a.dimensions()[0] == x.dimensions()[0] && a.dimensions()[1] == x.dimensions()[1] - static_cast<int>(a.dimensions()[2]),
                      "Input buffer `a` must have shape [B, T, order] when input buffer `x` has shape [B, T + order].");

        B = static_cast<int>(x.dimensions()[0]);
        x_len = static_cast<int>(x.dimensions()[1]);
        order = static_cast<int>(a.dimensions()[2]);
    }

    auto threads_per_block = order;

    std::copy_n(x.typed_data(), B * x_len, out->typed_data());
    allpole_kernel<float><<<B, threads_per_block, threads_per_block * sizeof(float), stream>>>(out->typed_data(), a.typed_data(), B, x_len - order, order);
    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess)
    {
        return ffi::Error::Internal(
            std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    allpole, allpole_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>() // stream
        .Arg<ffi::Buffer<ffi::F32>>()             // x
        .Arg<ffi::Buffer<ffi::F32>>()             // a
        .Ret<ffi::Buffer<ffi::F32>>()             // out
);

NB_MODULE(_filter_gpu, m)
{
    m.def("allpole", []()
          { return EncapsulateFfiCall(allpole); });
}