// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//


// declarations
#include "cudaTGaussian.h"
// dependencies
#include "cudaRandom.h"
// cuda utilities
// shared NTHREADS/IDIVUP/cudaCheckError/matrix_view_t/vector_view_t
#include "../support.h"

#include <pyre/cuda.h>
#include <curand_kernel.h>

// cuda kernels; defined here, ahead of the launcher functions below that instantiate and
// launch them (see cudaL2.cu/cudaGaussian.cu for why the ordering matters)
namespace cudaTGaussian_kernels {

    // one thread per sample: draw {theta[sample, idx_begin:idx_end]} from a Gaussian
    // truncated to the (already normalized, Phi-space) support [low, high)
    template <typename real_type>
    __global__ void
    _sample(curandState_t * curand_states, matrix_view_t<real_type, false> theta,
        const size_t idx_begin, const size_t idx_end,
        const real_type mean, const real_type sigma,
        const real_type low, const real_type high)
    {
        int sample = blockIdx.x*blockDim.x + threadIdx.x;
        auto samples = theta.packing().shape()[0];
        if (sample >= samples) return;

        unsigned long long seed = (unsigned long long) clock64();
        curand_init(seed, sample, 0, &curand_states[sample]);

        auto sqrt_two_sigma = sqrt(real_type{ 2 }) * sigma;
        auto range = high - low;

        for (auto i = idx_begin; i < idx_end; ++i) {
            auto temp = altar::cuda::distributions::curandUniform<real_type>(&curand_states[sample]) * range + low;
            theta[{ sample, static_cast<int>(i) }] = erfinv(real_type{ 2 } * temp - real_type{ 1 }) * sqrt_two_sigma + mean;
        }
    }

    // one thread per sample: add each sample's log pdf, summed over [idx_begin, idx_end),
    // into {probability[sample]}; {low}/{high} are the normalized (Phi-space) support bounds
    template <typename real_type>
    __global__ void
    _logpdf(matrix_view_t<real_type> theta, vector_view_t<real_type> probability,
        const size_t idx_begin, const size_t idx_end,
        const real_type mean, const real_type sigma,
        const real_type low, const real_type high)
    {
        int sample = blockIdx.x*blockDim.x + threadIdx.x;
        auto samples = theta.packing().shape()[0];
        if (sample >= samples) return;

        auto log_pdf = real_type{ 0 };
        auto c1 = -log(sigma * sqrt(real_type{ 2 } * PI) * (high - low));
        auto c2 = real_type{ 0.5 } / (sigma*sigma);

        for (auto i = idx_begin; i < idx_end; ++i) {
            auto mtmp = theta[{ sample, static_cast<int>(i) }] - mean;
            log_pdf += c1 - mtmp*mtmp*c2;
        }

        probability[{ sample }] += log_pdf;
    }

} // of namespace cudaTGaussian_kernels

// launch {cudaTGaussian_kernels::_sample}
template <typename real_type>
void altar::cuda::distributions::cudaTGaussian::
sample(matrix_view_t<real_type, false> theta,
    const size_t idx_begin, const size_t idx_end,
    const real_type mean, const real_type sigma,
    const real_type low, const real_type high,
    cudaStream_t stream)
{
    auto samples = theta.packing().shape()[0];
    auto blockSize = NTHREADS;
    auto gridSize = IDIVUP(samples, blockSize);

    curandState_t * curand_states;
    cudaSafeCall(cudaMalloc((void**)&curand_states, blockSize*gridSize*sizeof(curandState)));

    cudaTGaussian_kernels::_sample<real_type><<<gridSize, blockSize, 0, stream>>>(
        curand_states, theta, idx_begin, idx_end, mean, sigma, low, high);
    cudaCheckError("cudaTGaussian::sample error");

    cudaSafeCall(cudaFree(curand_states));
}

template void altar::cuda::distributions::cudaTGaussian::sample<float>(
    matrix_view_t<float, false>, const size_t, const size_t, const float, const float, const float, const float, cudaStream_t);
template void altar::cuda::distributions::cudaTGaussian::sample<double>(
    matrix_view_t<double, false>, const size_t, const size_t, const double, const double, const double, const double, cudaStream_t);


// launch {cudaTGaussian_kernels::_logpdf}
template <typename real_type>
void altar::cuda::distributions::cudaTGaussian::
logpdf(matrix_view_t<real_type> theta, vector_view_t<real_type> probability,
    const size_t idx_begin, const size_t idx_end,
    const real_type mean, const real_type sigma,
    const real_type low, const real_type high,
    cudaStream_t stream)
{
    auto samples = theta.packing().shape()[0];
    auto blockSize = NTHREADS;
    auto gridSize = IDIVUP(samples, blockSize);

    cudaTGaussian_kernels::_logpdf<real_type><<<gridSize, blockSize, 0, stream>>>(
        theta, probability, idx_begin, idx_end, mean, sigma, low, high);
    cudaCheckError("cudaTGaussian::logpdf error");
}

template void altar::cuda::distributions::cudaTGaussian::logpdf<float>(
    matrix_view_t<float>, vector_view_t<float>, const size_t, const size_t,
    const float, const float, const float, const float, cudaStream_t);
template void altar::cuda::distributions::cudaTGaussian::logpdf<double>(
    matrix_view_t<double>, vector_view_t<double>, const size_t, const size_t,
    const double, const double, const double, const double, cudaStream_t);

// end of file
