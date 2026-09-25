// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-2025 parasim inc
// (c) 2010-2025 california institute of technology
// all rights reserved
//
// Author(s): Lijun Zhu


// declarations
#include "cudaLogistic.h"
// dependencies
#include "cudaRandom.h"
// cuda utilities
// shared NTHREADS/IDIVUP/cudaCheckError/matrix_view_t/vector_view_t
#include "../support.h"

#include <pyre/cuda.h>
#include <curand_kernel.h>

// cuda kernels; defined here, ahead of the launcher functions below that instantiate and
// launch them (see cudaL2.cu/cudaGaussian.cu for why the ordering matters)
namespace cudaLogistic_kernels {

    // one thread per sample: draw {theta[sample, idx_begin:idx_end]} from the standard
    // logistic distribution, via the logit of a uniform draw
    template <typename real_type>
    __global__ void
    _sample(curandState_t * curand_states, matrix_view_t<real_type, false> theta,
        const size_t idx_begin, const size_t idx_end)
    {
        int sample = blockIdx.x*blockDim.x + threadIdx.x;
        auto samples = theta.packing().shape()[0];
        if (sample >= samples) return;

        unsigned long long seed = (unsigned long long) clock64();
        curand_init(seed, sample, 0, &curand_states[sample]);

        for (auto i = idx_begin; i < idx_end; ++i) {
            auto ran_num = altar::cuda::distributions::curandUniform<real_type>(&curand_states[sample]);
            theta[{ sample, static_cast<int>(i) }] = log(ran_num / (real_type{ 1 } - ran_num));
        }
    }

    // one thread per sample: add the standard logistic log pdf, summed over
    // [idx_begin, idx_end), into {probability[sample]}: log_pdf(x) = x - 2*log(1 + e^x)
    template <typename real_type>
    __global__ void
    _logpdf(matrix_view_t<real_type> theta, vector_view_t<real_type> probability,
        const size_t idx_begin, const size_t idx_end)
    {
        int sample = blockIdx.x*blockDim.x + threadIdx.x;
        auto samples = theta.packing().shape()[0];
        if (sample >= samples) return;

        auto log_pdf = real_type{ 0 };
        for (auto i = idx_begin; i < idx_end; ++i) {
            auto theta_i = theta[{ sample, static_cast<int>(i) }];
            log_pdf += theta_i - real_type{ 2 } * log(real_type{ 1 } + exp(theta_i));
        }

        probability[{ sample }] += log_pdf;
    }

    // one thread per sample: fill {probability[sample, idx_begin:idx_end]} with
    // d/dx [x - 2*log(1 + e^x)] = 2/(1 + e^x) - 1
    template <typename real_type>
    __global__ void
    _logpdfgradient(matrix_view_t<real_type> theta, matrix_view_t<real_type, false> probability,
        const size_t idx_begin, const size_t idx_end)
    {
        int sample = blockIdx.x*blockDim.x + threadIdx.x;
        auto samples = theta.packing().shape()[0];
        if (sample >= samples) return;

        for (auto i = idx_begin; i < idx_end; ++i) {
            auto theta_i = theta[{ sample, static_cast<int>(i) }];
            probability[{ sample, static_cast<int>(i) }] = real_type{ 2 } / (real_type{ 1 } + exp(theta_i)) - real_type{ 1 };
        }
    }

} // of namespace cudaLogistic_kernels

// launch {cudaLogistic_kernels::_sample}
template <typename real_type>
void altar::cuda::distributions::cudaLogistic::
sample(matrix_view_t<real_type, false> theta,
    const size_t idx_begin, const size_t idx_end,
    cudaStream_t stream)
{
    auto samples = theta.packing().shape()[0];
    auto blockSize = NTHREADS;
    auto gridSize = IDIVUP(samples, blockSize);

    curandState_t * curand_states;
    cudaSafeCall(cudaMalloc((void**)&curand_states, blockSize*gridSize*sizeof(curandState)));

    cudaLogistic_kernels::_sample<real_type><<<gridSize, blockSize, 0, stream>>>(
        curand_states, theta, idx_begin, idx_end);
    cudaCheckError("cudaLogistic::sample error");

    cudaSafeCall(cudaFree(curand_states));
}

template void altar::cuda::distributions::cudaLogistic::sample<float>(
    matrix_view_t<float, false>, const size_t, const size_t, cudaStream_t);
template void altar::cuda::distributions::cudaLogistic::sample<double>(
    matrix_view_t<double, false>, const size_t, const size_t, cudaStream_t);


// launch {cudaLogistic_kernels::_logpdf}
template <typename real_type>
void altar::cuda::distributions::cudaLogistic::
logpdf(matrix_view_t<real_type> theta, vector_view_t<real_type> probability,
    const size_t idx_begin, const size_t idx_end,
    cudaStream_t stream)
{
    auto samples = theta.packing().shape()[0];
    auto blockSize = NTHREADS;
    auto gridSize = IDIVUP(samples, blockSize);

    cudaLogistic_kernels::_logpdf<real_type><<<gridSize, blockSize, 0, stream>>>(
        theta, probability, idx_begin, idx_end);
    cudaCheckError("cudaLogistic::logpdf error");
}

template void altar::cuda::distributions::cudaLogistic::logpdf<float>(
    matrix_view_t<float>, vector_view_t<float>, const size_t, const size_t, cudaStream_t);
template void altar::cuda::distributions::cudaLogistic::logpdf<double>(
    matrix_view_t<double>, vector_view_t<double>, const size_t, const size_t, cudaStream_t);


// launch {cudaLogistic_kernels::_logpdfgradient}
template <typename real_type>
void altar::cuda::distributions::cudaLogistic::
logpdfgradient(matrix_view_t<real_type> theta, matrix_view_t<real_type, false> probability,
    const size_t idx_begin, const size_t idx_end,
    cudaStream_t stream)
{
    auto samples = theta.packing().shape()[0];
    auto blockSize = NTHREADS;
    auto gridSize = IDIVUP(samples, blockSize);

    cudaLogistic_kernels::_logpdfgradient<real_type><<<gridSize, blockSize, 0, stream>>>(
        theta, probability, idx_begin, idx_end);
    cudaCheckError("cudaLogistic::logpdfgradient error");
}

template void altar::cuda::distributions::cudaLogistic::logpdfgradient<float>(
    matrix_view_t<float>, matrix_view_t<float, false>, const size_t, const size_t, cudaStream_t);
template void altar::cuda::distributions::cudaLogistic::logpdfgradient<double>(
    matrix_view_t<double>, matrix_view_t<double, false>, const size_t, const size_t, cudaStream_t);

// end of file
