// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-2025 parasim inc
// (c) 2010-2025 california institute of technology
// all rights reserved
//
// Author(s): Hailiang Zhang, Lijun Zhu


// declarations
#include "cudaGaussian.h"
// dependencies
#include "cudaRandom.h"
// cuda utilities
// shared NTHREADS/IDIVUP/cudaCheckError/matrix_view_t/vector_view_t
#include "../support.h"

#include <pyre/cuda.h>
#include <curand_kernel.h>

// cuda kernels; defined here, ahead of the launcher functions below that instantiate and
// launch them -- nvcc's handling of a __global__ function template instantiated from a
// launch that only sees a forward declaration (body defined later in the same file) is not
// reliable the way a plain host template's deferred instantiation is, so the definition has
// to come first
namespace cudaGaussian_kernels {

    // one thread per sample: draw {theta[sample, idx_begin:idx_end]} from N(mean, sigma^2)
    template <typename real_type>
    __global__ void
    _sample(curandState_t * curand_states, matrix_view_t<real_type, false> theta,
        const size_t idx_begin, const size_t idx_end,
        const real_type mean, const real_type sigma)
    {
        int sample = blockIdx.x*blockDim.x + threadIdx.x;
        auto samples = theta.packing().shape()[0];
        if (sample >= samples) return;

        // one curand state per thread
        unsigned long long seed = (unsigned long long) clock64();
        curand_init(seed, sample, 0, &curand_states[sample]);

        for (auto i = idx_begin; i < idx_end; ++i) {
            theta[{ sample, static_cast<int>(i) }] =
                altar::cuda::distributions::curandNormal<real_type>(&curand_states[sample]) * sigma + mean;
        }
    }

    // one thread per sample: add the log pdf, summed over [idx_begin, idx_end), into
    // {probability[sample]}
    template <typename real_type>
    __global__ void
    _logpdf(matrix_view_t<real_type> theta, vector_view_t<real_type> probability,
        const size_t idx_begin, const size_t idx_end,
        const real_type mean, const real_type sigma)
    {
        int sample = blockIdx.x*blockDim.x + threadIdx.x;
        auto samples = theta.packing().shape()[0];
        if (sample >= samples) return;

        auto log_pdf = real_type{ 0 };
        auto c1 = -log(sigma * sqrt(real_type{ 2 } * PI));
        auto c2 = real_type{ 0.5 } / (sigma*sigma);

        for (auto i = idx_begin; i < idx_end; ++i) {
            auto mtmp = theta[{ sample, static_cast<int>(i) }] - mean;
            log_pdf += c1 - mtmp*mtmp*c2;
        }

        probability[{ sample }] += log_pdf;
    }

    // one thread per sample: add the log pdf gradient with respect to parameter {index}
    // into {probability[sample]}; a no-op when {index} falls outside [idx_begin, idx_end)
    template <typename real_type>
    __global__ void
    _logpdfgradient_index(matrix_view_t<real_type> theta, vector_view_t<real_type> probability,
        const size_t idx_begin, const size_t idx_end, const int index,
        const real_type mean, const real_type sigma)
    {
        int sample = blockIdx.x*blockDim.x + threadIdx.x;
        auto samples = theta.packing().shape()[0];
        if (sample >= samples) return;
        if (index < static_cast<int>(idx_begin) || index >= static_cast<int>(idx_end)) return;

        auto log_pdf_gradient = (mean - theta[{ sample, index }]) / (sigma*sigma);
        probability[{ sample }] += log_pdf_gradient;
    }

    // one thread per sample: fill {probability[sample, idx_begin:idx_end]} with the log pdf
    // gradient with respect to every parameter in that range (a plain assignment, not an
    // accumulation, unlike {_logpdfgradient_index} above)
    template <typename real_type>
    __global__ void
    _logpdfgradient(matrix_view_t<real_type> theta, matrix_view_t<real_type, false> probability,
        const size_t idx_begin, const size_t idx_end,
        const real_type mean, const real_type sigma)
    {
        int sample = blockIdx.x*blockDim.x + threadIdx.x;
        auto samples = theta.packing().shape()[0];
        if (sample >= samples) return;

        auto c2 = real_type{ 1 } / (sigma*sigma);
        for (auto i = idx_begin; i < idx_end; ++i) {
            probability[{ sample, static_cast<int>(i) }] = (mean - theta[{ sample, static_cast<int>(i) }]) * c2;
        }
    }

} // of namespace cudaGaussian_kernels

// launch {cudaGaussian_kernels::_sample}
template <typename real_type>
void altar::cuda::distributions::cudaGaussian::
sample(matrix_view_t<real_type, false> theta,
    const size_t idx_begin, const size_t idx_end,
    const real_type mean, const real_type sigma,
    cudaStream_t stream)
{
    auto samples = theta.packing().shape()[0];

    auto blockSize = NTHREADS;
    auto gridSize = IDIVUP(samples, blockSize);

    curandState_t * curand_states;
    cudaSafeCall(cudaMalloc((void**)&curand_states, blockSize*gridSize*sizeof(curandState)));

    cudaGaussian_kernels::_sample<real_type><<<gridSize, blockSize, 0, stream>>>(
        curand_states, theta, idx_begin, idx_end, mean, sigma);
    cudaCheckError("cudaGaussian::sample error");

    cudaSafeCall(cudaFree(curand_states));
}

template void altar::cuda::distributions::cudaGaussian::sample<float>(
    matrix_view_t<float, false>, const size_t, const size_t, const float, const float, cudaStream_t);
template void altar::cuda::distributions::cudaGaussian::sample<double>(
    matrix_view_t<double, false>, const size_t, const size_t, const double, const double, cudaStream_t);


// launch {cudaGaussian_kernels::_logpdf}
template <typename real_type>
void altar::cuda::distributions::cudaGaussian::
logpdf(matrix_view_t<real_type> theta, vector_view_t<real_type> probability,
    const size_t idx_begin, const size_t idx_end,
    const real_type mean, const real_type sigma,
    cudaStream_t stream)
{
    auto samples = theta.packing().shape()[0];
    auto blockSize = NTHREADS;
    auto gridSize = IDIVUP(samples, blockSize);

    cudaGaussian_kernels::_logpdf<real_type><<<gridSize, blockSize, 0, stream>>>(
        theta, probability, idx_begin, idx_end, mean, sigma);
    cudaCheckError("cudaGaussian::logpdf error");
}

template void altar::cuda::distributions::cudaGaussian::logpdf<float>(
    matrix_view_t<float>, vector_view_t<float>, const size_t, const size_t, const float, const float, cudaStream_t);
template void altar::cuda::distributions::cudaGaussian::logpdf<double>(
    matrix_view_t<double>, vector_view_t<double>, const size_t, const size_t, const double, const double, cudaStream_t);


// launch {cudaGaussian_kernels::_logpdfgradient_index}
template <typename real_type>
void altar::cuda::distributions::cudaGaussian::
logpdfgradient_i(matrix_view_t<real_type> theta, vector_view_t<real_type> probability,
    const size_t idx_begin, const size_t idx_end, const size_t index,
    const real_type mean, const real_type sigma,
    cudaStream_t stream)
{
    auto samples = theta.packing().shape()[0];
    auto blockSize = NTHREADS;
    auto gridSize = IDIVUP(samples, blockSize);

    cudaGaussian_kernels::_logpdfgradient_index<real_type><<<gridSize, blockSize, 0, stream>>>(
        theta, probability, idx_begin, idx_end, static_cast<int>(index), mean, sigma);
    cudaCheckError("cudaGaussian::logpdfgradient_i error");
}

template void altar::cuda::distributions::cudaGaussian::logpdfgradient_i<float>(
    matrix_view_t<float>, vector_view_t<float>, const size_t, const size_t, const size_t,
    const float, const float, cudaStream_t);
template void altar::cuda::distributions::cudaGaussian::logpdfgradient_i<double>(
    matrix_view_t<double>, vector_view_t<double>, const size_t, const size_t, const size_t,
    const double, const double, cudaStream_t);


// launch {cudaGaussian_kernels::_logpdfgradient}
template <typename real_type>
void altar::cuda::distributions::cudaGaussian::
logpdfgradient(matrix_view_t<real_type> theta, matrix_view_t<real_type, false> probability,
    const size_t idx_begin, const size_t idx_end,
    const real_type mean, const real_type sigma,
    cudaStream_t stream)
{
    auto samples = theta.packing().shape()[0];
    auto blockSize = NTHREADS;
    auto gridSize = IDIVUP(samples, blockSize);

    cudaGaussian_kernels::_logpdfgradient<real_type><<<gridSize, blockSize, 0, stream>>>(
        theta, probability, idx_begin, idx_end, mean, sigma);
    cudaCheckError("cudaGaussian::logpdfgradient error");
}

template void altar::cuda::distributions::cudaGaussian::logpdfgradient<float>(
    matrix_view_t<float>, matrix_view_t<float, false>, const size_t, const size_t, const float, const float, cudaStream_t);
template void altar::cuda::distributions::cudaGaussian::logpdfgradient<double>(
    matrix_view_t<double>, matrix_view_t<double, false>, const size_t, const size_t, const double, const double, cudaStream_t);

// end of file
