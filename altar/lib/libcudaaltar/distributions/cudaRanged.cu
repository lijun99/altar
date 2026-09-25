// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//
// hailiang zhang

// declarations
#include "cudaRanged.h"
// cuda utilities
// shared NTHREADS/IDIVUP/cudaCheckError/matrix_view_t/vector_view_t
#include "../support.h"

#include <pyre/cuda.h>

// cuda kernels; defined here, ahead of the launcher functions below that instantiate and
// launch them (see cudaL2.cu/cudaGaussian.cu for why the ordering matters)
namespace cudaRanged_kernels {

    // one thread per sample: flag {invalid[sample] = 1} if any of
    // {theta[sample, idx_begin:idx_end]} falls outside [low, high]; a sample already flagged
    // is left alone
    template <typename real_type>
    __global__ void
    _verify(matrix_view_t<real_type> theta, vector_view_t<int> invalid,
        const size_t idx_begin, const size_t idx_end,
        const real_type low, const real_type high)
    {
        int sample = blockIdx.x*blockDim.x + threadIdx.x;
        auto samples = theta.packing().shape()[0];
        if (sample >= samples) return;
        if (invalid[{ sample }]) return;

        for (auto i = idx_begin; i < idx_end; ++i) {
            auto value = theta[{ sample, static_cast<int>(i) }];
            if (value < low || value > high) {
                invalid[{ sample }] = 1;
                return;
            }
        }
    }

    // the per-parameter-bounds counterpart of {_verify}
    template <typename real_type>
    __global__ void
    _verify_unique(matrix_view_t<real_type> theta, vector_view_t<int> invalid,
        const size_t idx_begin, const size_t idx_end,
        vector_view_t<real_type, true> low, vector_view_t<real_type, true> high)
    {
        int sample = blockIdx.x*blockDim.x + threadIdx.x;
        auto samples = theta.packing().shape()[0];
        if (sample >= samples) return;
        if (invalid[{ sample }]) return;

        int j = 0;
        for (auto i = idx_begin; i < idx_end; ++i, ++j) {
            auto value = theta[{ sample, static_cast<int>(i) }];
            if (value < low[{ j }] || value > high[{ j }]) {
                invalid[{ sample }] = 1;
                return;
            }
        }
    }

    // one thread per sample: clamp {theta[sample, idx_begin:idx_end]} into [low, high]
    template <typename real_type>
    __global__ void
    _constrain(matrix_view_t<real_type, false> theta,
        const size_t idx_begin, const size_t idx_end,
        const real_type low, const real_type high)
    {
        int sample = blockIdx.x*blockDim.x + threadIdx.x;
        auto samples = theta.packing().shape()[0];
        if (sample >= samples) return;

        for (auto i = idx_begin; i < idx_end; ++i) {
            auto value = theta[{ sample, static_cast<int>(i) }];
            if (value < low) {
                theta[{ sample, static_cast<int>(i) }] = low;
            } else if (value > high) {
                theta[{ sample, static_cast<int>(i) }] = high;
            }
        }
    }

} // of namespace cudaRanged_kernels

// launch {cudaRanged_kernels::_verify}
template <typename real_type>
void altar::cuda::distributions::cudaRanged::
verify(matrix_view_t<real_type> theta, vector_view_t<int> invalid,
    const size_t idx_begin, const size_t idx_end,
    const real_type low, const real_type high,
    cudaStream_t stream)
{
    auto samples = theta.packing().shape()[0];
    auto blockSize = NTHREADS;
    auto gridSize = IDIVUP(samples, blockSize);

    cudaRanged_kernels::_verify<real_type><<<gridSize, blockSize, 0, stream>>>(
        theta, invalid, idx_begin, idx_end, low, high);
    cudaCheckError("cudaRanged::verify error");
}

template void altar::cuda::distributions::cudaRanged::verify<float>(
    matrix_view_t<float>, vector_view_t<int>, const size_t, const size_t, const float, const float, cudaStream_t);
template void altar::cuda::distributions::cudaRanged::verify<double>(
    matrix_view_t<double>, vector_view_t<int>, const size_t, const size_t, const double, const double, cudaStream_t);


// launch {cudaRanged_kernels::_verify_unique}
template <typename real_type>
void altar::cuda::distributions::cudaRanged::
verify_unique(matrix_view_t<real_type> theta, vector_view_t<int> invalid,
    const size_t idx_begin, const size_t idx_end,
    vector_view_t<real_type, true> low, vector_view_t<real_type, true> high,
    cudaStream_t stream)
{
    auto samples = theta.packing().shape()[0];
    auto blockSize = NTHREADS;
    auto gridSize = IDIVUP(samples, blockSize);

    cudaRanged_kernels::_verify_unique<real_type><<<gridSize, blockSize, 0, stream>>>(
        theta, invalid, idx_begin, idx_end, low, high);
    cudaCheckError("cudaRanged::verify_unique error");
}

template void altar::cuda::distributions::cudaRanged::verify_unique<float>(
    matrix_view_t<float>, vector_view_t<int>, const size_t, const size_t,
    vector_view_t<float, true>, vector_view_t<float, true>, cudaStream_t);
template void altar::cuda::distributions::cudaRanged::verify_unique<double>(
    matrix_view_t<double>, vector_view_t<int>, const size_t, const size_t,
    vector_view_t<double, true>, vector_view_t<double, true>, cudaStream_t);


// launch {cudaRanged_kernels::_constrain}
template <typename real_type>
void altar::cuda::distributions::cudaRanged::
constrain(matrix_view_t<real_type, false> theta,
    const size_t idx_begin, const size_t idx_end,
    const real_type low, const real_type high,
    cudaStream_t stream)
{
    auto samples = theta.packing().shape()[0];
    auto blockSize = NTHREADS;
    auto gridSize = IDIVUP(samples, blockSize);

    cudaRanged_kernels::_constrain<real_type><<<gridSize, blockSize, 0, stream>>>(
        theta, idx_begin, idx_end, low, high);
    cudaCheckError("cudaRanged::constrain error");
}

template void altar::cuda::distributions::cudaRanged::constrain<float>(
    matrix_view_t<float, false>, const size_t, const size_t, const float, const float, cudaStream_t);
template void altar::cuda::distributions::cudaRanged::constrain<double>(
    matrix_view_t<double, false>, const size_t, const size_t, const double, const double, cudaStream_t);

// end of file
