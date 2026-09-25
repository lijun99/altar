// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//
// hailiang zhang

// declarations
#include "cudaUniform.h"
// dependencies
#include "cudaRandom.h"
// cuda utilities
// shared NTHREADS/IDIVUP/cudaCheckError/matrix_view_t/vector_view_t
#include "../support.h"

#include <pyre/cuda.h>
#include <curand_kernel.h>

// cuda kernels; defined here, ahead of the launcher functions below that instantiate and
// launch them (see cudaL2.cu/cudaGaussian.cu for why the ordering matters)
namespace cudaUniform_kernels {

    // one thread per sample: draw {theta[sample, idx_begin:idx_end]} ~ U[low, high)
    template <typename real_type>
    __global__ void
    _sample(curandState_t * curand_states, matrix_view_t<real_type, false> theta,
        const size_t idx_begin, const size_t idx_end,
        const real_type low, const real_type high)
    {
        int sample = blockIdx.x*blockDim.x + threadIdx.x;
        auto samples = theta.packing().shape()[0];
        if (sample >= samples) return;

        unsigned long long seed = (unsigned long long) clock64();
        curand_init(seed, sample, 0, &curand_states[sample]);

        auto range = high - low;
        for (auto i = idx_begin; i < idx_end; ++i) {
            theta[{ sample, static_cast<int>(i) }] =
                altar::cuda::distributions::curandUniform<real_type>(&curand_states[sample]) * range + low;
        }
    }

    // the per-parameter-bounds counterpart of {_sample}: {low[j]}/{high[j]} apply to
    // parameter {idx_begin + j}
    template <typename real_type>
    __global__ void
    _sample_unique(curandState_t * curand_states, matrix_view_t<real_type, false> theta,
        const size_t idx_begin, const size_t idx_end,
        vector_view_t<real_type, true> low, vector_view_t<real_type, true> high)
    {
        int sample = blockIdx.x*blockDim.x + threadIdx.x;
        auto samples = theta.packing().shape()[0];
        if (sample >= samples) return;

        unsigned long long seed = (unsigned long long) clock64();
        curand_init(seed, sample, 0, &curand_states[sample]);

        int j = 0;
        for (auto i = idx_begin; i < idx_end; ++i, ++j) {
            auto range = high[{ j }] - low[{ j }];
            theta[{ sample, static_cast<int>(i) }] =
                altar::cuda::distributions::curandUniform<real_type>(&curand_states[sample]) * range + low[{ j }];
        }
    }

    // one thread per sample: add log(1 / (high - low)) * (idx_end - idx_begin) into
    // {probability[sample]} -- constant per sample, since a uniform's log pdf doesn't depend
    // on {theta} itself, only on the range
    template <typename real_type>
    __global__ void
    _logpdf(matrix_view_t<real_type> theta, vector_view_t<real_type> probability,
        const size_t idx_begin, const size_t idx_end,
        const real_type low, const real_type high)
    {
        int sample = blockIdx.x*blockDim.x + threadIdx.x;
        auto samples = theta.packing().shape()[0];
        if (sample >= samples) return;

        auto log_pdf = -log(high - low) * static_cast<real_type>(idx_end - idx_begin);
        probability[{ sample }] += log_pdf;
    }

    // the per-parameter-bounds counterpart of {_logpdf}
    template <typename real_type>
    __global__ void
    _logpdf_unique(matrix_view_t<real_type> theta, vector_view_t<real_type> probability,
        const size_t idx_begin, const size_t idx_end,
        vector_view_t<real_type, true> low, vector_view_t<real_type, true> high)
    {
        int sample = blockIdx.x*blockDim.x + threadIdx.x;
        auto samples = theta.packing().shape()[0];
        if (sample >= samples) return;

        int j = 0;
        for (auto i = idx_begin; i < idx_end; ++i, ++j) {
            auto log_pdf = -log(high[{ j }] - low[{ j }]);
            probability[{ sample }] += log_pdf;
        }
    }

} // of namespace cudaUniform_kernels

// launch {cudaUniform_kernels::_sample}
template <typename real_type>
void altar::cuda::distributions::cudaUniform::
sample(matrix_view_t<real_type, false> theta,
    const size_t idx_begin, const size_t idx_end,
    const real_type low, const real_type high,
    cudaStream_t stream)
{
    auto samples = theta.packing().shape()[0];
    auto blockSize = NTHREADS;
    auto gridSize = IDIVUP(samples, blockSize);

    curandState_t * curand_states;
    cudaSafeCall(cudaMalloc((void**)&curand_states, blockSize*gridSize*sizeof(curandState)));

    cudaUniform_kernels::_sample<real_type><<<gridSize, blockSize, 0, stream>>>(
        curand_states, theta, idx_begin, idx_end, low, high);
    cudaCheckError("cudaUniform::sample error");

    cudaSafeCall(cudaFree(curand_states));
}

template void altar::cuda::distributions::cudaUniform::sample<float>(
    matrix_view_t<float, false>, const size_t, const size_t, const float, const float, cudaStream_t);
template void altar::cuda::distributions::cudaUniform::sample<double>(
    matrix_view_t<double, false>, const size_t, const size_t, const double, const double, cudaStream_t);


// launch {cudaUniform_kernels::_sample_unique}
template <typename real_type>
void altar::cuda::distributions::cudaUniform::
sample_unique(matrix_view_t<real_type, false> theta,
    const size_t idx_begin, const size_t idx_end,
    vector_view_t<real_type, true> low, vector_view_t<real_type, true> high,
    cudaStream_t stream)
{
    auto samples = theta.packing().shape()[0];
    auto blockSize = NTHREADS;
    auto gridSize = IDIVUP(samples, blockSize);

    curandState_t * curand_states;
    cudaSafeCall(cudaMalloc((void**)&curand_states, blockSize*gridSize*sizeof(curandState)));

    cudaUniform_kernels::_sample_unique<real_type><<<gridSize, blockSize, 0, stream>>>(
        curand_states, theta, idx_begin, idx_end, low, high);
    cudaCheckError("cudaUniform::sample_unique error");

    cudaSafeCall(cudaFree(curand_states));
}

template void altar::cuda::distributions::cudaUniform::sample_unique<float>(
    matrix_view_t<float, false>, const size_t, const size_t,
    vector_view_t<float, true>, vector_view_t<float, true>, cudaStream_t);
template void altar::cuda::distributions::cudaUniform::sample_unique<double>(
    matrix_view_t<double, false>, const size_t, const size_t,
    vector_view_t<double, true>, vector_view_t<double, true>, cudaStream_t);


// launch {cudaUniform_kernels::_logpdf}
template <typename real_type>
void altar::cuda::distributions::cudaUniform::
logpdf(matrix_view_t<real_type> theta, vector_view_t<real_type> probability,
    const size_t idx_begin, const size_t idx_end,
    const real_type low, const real_type high,
    cudaStream_t stream)
{
    auto samples = theta.packing().shape()[0];
    auto blockSize = NTHREADS;
    auto gridSize = IDIVUP(samples, blockSize);

    cudaUniform_kernels::_logpdf<real_type><<<gridSize, blockSize, 0, stream>>>(
        theta, probability, idx_begin, idx_end, low, high);
    cudaCheckError("cudaUniform::logpdf error");
}

template void altar::cuda::distributions::cudaUniform::logpdf<float>(
    matrix_view_t<float>, vector_view_t<float>, const size_t, const size_t, const float, const float, cudaStream_t);
template void altar::cuda::distributions::cudaUniform::logpdf<double>(
    matrix_view_t<double>, vector_view_t<double>, const size_t, const size_t, const double, const double, cudaStream_t);


// launch {cudaUniform_kernels::_logpdf_unique}
template <typename real_type>
void altar::cuda::distributions::cudaUniform::
logpdf_unique(matrix_view_t<real_type> theta, vector_view_t<real_type> probability,
    const size_t idx_begin, const size_t idx_end,
    vector_view_t<real_type, true> low, vector_view_t<real_type, true> high,
    cudaStream_t stream)
{
    auto samples = theta.packing().shape()[0];
    auto blockSize = NTHREADS;
    auto gridSize = IDIVUP(samples, blockSize);

    cudaUniform_kernels::_logpdf_unique<real_type><<<gridSize, blockSize, 0, stream>>>(
        theta, probability, idx_begin, idx_end, low, high);
    cudaCheckError("cudaUniform::logpdf_unique error");
}

template void altar::cuda::distributions::cudaUniform::logpdf_unique<float>(
    matrix_view_t<float>, vector_view_t<float>, const size_t, const size_t,
    vector_view_t<float, true>, vector_view_t<float, true>, cudaStream_t);
template void altar::cuda::distributions::cudaUniform::logpdf_unique<double>(
    matrix_view_t<double>, vector_view_t<double>, const size_t, const size_t,
    vector_view_t<double, true>, vector_view_t<double, true>, cudaStream_t);

// end of file
