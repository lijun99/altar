// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-2025 parasim inc
// (c) 2010-2025 california institute of technology
// all rights reserved
//
// Author(s):  Lijun Zhu


// declarations
#include "cudaTGaussianLogit.h"
// cuda utilities
#include <pyre/cuda.h>
#include <curand_kernel.h>

// cuda kernel declarations
namespace cudaTGaussianLogit_kernels {

    template<typename real_type>
    __device__ real_type gaussian_cdf( const real_type theta,
        const real_type mean, const real_type sigma);

    template<typename real_type>
    __device__ real_type gaussian_cdfinv( const real_type p,
        const real_type mean, const real_type sigma);

    template<typename real_type>
    __global__ void _tosampling(real_type * const theta,
        const size_t samples, const size_t parameters,
        const size_t idx_begin, const size_t idx_end,
        const real_type mean, const real_type sigma,
        const real_type low, const real_type high);

    template<typename real_type>
    __global__ void _tophysical(real_type * const theta,
        const size_t samples, const size_t parameters,
        const size_t idx_begin, const size_t idx_end,
        const real_type mean, const real_type sigma,
        const real_type low, const real_type high);

} // of namespace cudaTGaussianLogit_kernels


// transform logistic variable to bounded
template <typename real_type>
void altar::cuda::distributions::cudaTGaussianLogit::
tophysical(real_type * const theta,
        const size_t samples, const size_t parameters,
        const size_t idx_begin, const size_t idx_end,
        const real_type mean, const real_type sigma,
        const real_type low_cdf, const real_type high_cdf,
        cudaStream_t stream)
{
    int blockSize = NTHREADS;
    int gridSize = IDIVUP(samples, blockSize);
    // call cuda kernels
    cudaTGaussianLogit_kernels::_tophysical<real_type><<<gridSize, blockSize, 0, stream>>>(
        theta, samples, parameters, idx_begin, idx_end, mean, sigma, low_cdf, high_cdf);
    cudaCheckError("cudaTGaussianLogit:: tophysical error");
}

// explicit instantiation
template void altar::cuda::distributions::cudaTGaussianLogit::tophysical<float>(float * const, const size_t, const size_t,
                    const size_t, const size_t, const float, const float, const float, const float, cudaStream_t);
template void altar::cuda::distributions::cudaTGaussianLogit::tophysical<double>(double * const, const size_t, const size_t,
                    const size_t, const size_t, const double, const double, const double, const double, cudaStream_t);

// transform logistic variable to bounded
template <typename real_type>
void altar::cuda::distributions::cudaTGaussianLogit::
tosampling(real_type * const theta,
        const size_t samples, const size_t parameters,
        const size_t idx_begin, const size_t idx_end,
        const real_type mean, const real_type sigma,
        const real_type low_cdf, const real_type high_cdf,
        cudaStream_t stream)
{
    int blockSize = NTHREADS;
    int gridSize = IDIVUP(samples, blockSize);
    // call cuda kernels
    cudaTGaussianLogit_kernels::_tosampling<real_type><<<gridSize, blockSize, 0, stream>>>(
        theta, samples, parameters, idx_begin, idx_end, mean, sigma, low_cdf, high_cdf);
    cudaCheckError("cudaTGaussianLogit:: tosampling error");
}

// explicit instantiation
template void altar::cuda::distributions::cudaTGaussianLogit::tosampling<float>(float * const, const size_t, const size_t,
                    const size_t, const size_t, const float, const float, const float, const float, cudaStream_t);
template void altar::cuda::distributions::cudaTGaussianLogit::tosampling<double>(double * const, const size_t, const size_t,
                    const size_t, const size_t, const double, const double, const double, const double, cudaStream_t);

// put explicit specialization in a namespace for compatibility with gcc6
namespace cudaTGaussianLogit_kernels {

// untruncated gaussian cdf function
template<typename real_type>
__device__ real_type gaussian_cdf( const real_type theta,
    const real_type mean, const real_type sigma)
{
    return 0.5*(1+erf((theta-mean)/(sqrt(2.0)*sigma)));
}

// inverse of untruncated gaussian cdf function
template<typename real_type>
__device__ real_type gaussian_cdfinv(const real_type p,
    const real_type mean, const real_type sigma)
{
    return mean+sqrt(2.0)*sigma*erfinv(2.0*p-1.0);
}

// tosampling kernel (logit)
template <typename real_type>
__global__ void
_tosampling(real_type * const theta, const size_t samples, const size_t parameters,
        const size_t idx_begin, const size_t idx_end,
        const real_type mean, const real_type sigma,
        const real_type low_cdf, const real_type high_cdf)
{
    // get the thread/sample id
    int sample = blockIdx.x*blockDim.x + threadIdx.x;
    if (sample >= samples) return;
    // get the starting pointer for this sample
    real_type * theta_sample = theta + sample*parameters;

    real_type diff_cdf = high_cdf-low_cdf;

    // check each parameter
    for (int i=idx_begin; i<idx_end; ++i)
    {
        real_type u = (gaussian_cdf(theta_sample[i], mean, sigma)-low_cdf)/diff_cdf;
        theta_sample[i] = log(u/(1.0-u));
    }
}

// tophysical kernel (expit/sigmoid)
template <typename real_type>
__global__ void
_tophysical(real_type * const theta, const size_t samples, const size_t parameters,
        const size_t idx_begin, const size_t idx_end,
        const real_type mean, const real_type sigma,
        const real_type low_cdf, const real_type high_cdf)
{
    // get the thread/sample id
    int sample = blockIdx.x*blockDim.x + threadIdx.x;
    if (sample >= samples) return;
    // get the starting pointer for this sample
    real_type * theta_sample = theta + sample*parameters;

    real_type diff_cdf = high_cdf-low_cdf;

    // check each parameter
    for (int i=idx_begin; i<idx_end; ++i)
    {
        real_type p = low_cdf + diff_cdf/(1.0+exp(-theta_sample[i]));
        theta_sample[i] = gaussian_cdfinv(p, mean, sigma);
    }
}


} // of namespace cudaTGaussianLogit_kernels

// end of file
