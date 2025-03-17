// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-2025 parasim inc
// (c) 2010-2025 california institute of technology
// all rights reserved
//
// Author(s):  Lijun Zhu


// declarations
#include "cudaUniformLogit.h"
// dependencies
#include "cudaRandom.h"
#include "cudaUniform.h"
// cuda utilities
#include <pyre/cuda.h>
#include <curand_kernel.h>

// cuda kernel declarations
namespace cudaUniformLogit_kernels {

    template<typename real_type>
    __global__ void _tosampling(real_type * const theta,
        const size_t samples, const size_t parameters,
        const size_t idx_begin, const size_t idx_end,
        const real_type low, const real_type high);

    template<typename real_type>
    __global__ void _tophysical(real_type * const theta,
        const size_t samples, const size_t parameters,
        const size_t idx_begin, const size_t idx_end,
        const real_type low, const real_type high);

} // of namespace cudaUniformLogit_kernels


// transform logistic variable to bounded
template <typename real_type>
void altar::cuda::distributions::cudaUniformLogit::
tophysical(real_type * const theta,
        const size_t samples, const size_t parameters,
        const size_t idx_begin, const size_t idx_end,
        const real_type low, const real_type high,
        cudaStream_t stream)
{
    int blockSize = NTHREADS;
    int gridSize = IDIVUP(samples, blockSize);
    // call cuda kernels
    cudaUniformLogit_kernels::_tophysical<real_type><<<gridSize, blockSize, 0, stream>>>(
        theta, samples, parameters, idx_begin, idx_end, low, high);
    cudaCheckError("cudaUniformLogit:: tophysical error");
}

// explicit instantiation
template void altar::cuda::distributions::cudaUniformLogit::tophysical<float>(float * const, const size_t, const size_t,
                    const size_t, const size_t, const float, const float, cudaStream_t);
template void altar::cuda::distributions::cudaUniformLogit::tophysical<double>(double * const, const size_t, const size_t,
                    const size_t, const size_t, const double, const double, cudaStream_t);

// transform logistic variable to bounded
template <typename real_type>
void altar::cuda::distributions::cudaUniformLogit::
tosampling(real_type * const theta,
        const size_t samples, const size_t parameters,
        const size_t idx_begin, const size_t idx_end,
        const real_type low, const real_type high,
        cudaStream_t stream)
{
    int blockSize = NTHREADS;
    int gridSize = IDIVUP(samples, blockSize);
    // call cuda kernels
    cudaUniformLogit_kernels::_tosampling<real_type><<<gridSize, blockSize, 0, stream>>>(
        theta, samples, parameters, idx_begin, idx_end, low, high);
    cudaCheckError("cudaUniformLogit:: tosampling error");
}

// explicit instantiation
template void altar::cuda::distributions::cudaUniformLogit::tosampling<float>(float * const, const size_t, const size_t,
                    const size_t, const size_t, const float, const float, cudaStream_t);
template void altar::cuda::distributions::cudaUniformLogit::tosampling<double>(double * const, const size_t, const size_t,
                    const size_t, const size_t, const double, const double, cudaStream_t);


//random_generation_kernel
// double precision version

namespace cudaUniformLogit_kernels {

// tosampling kernel (logit)
template <typename real_type>
__global__ void
_tosampling(real_type * const theta, const size_t samples, const size_t parameters,
        const size_t idx_begin, const size_t idx_end, const real_type low, const real_type high)
{
    // get the thread/sample id
    int sample = blockIdx.x*blockDim.x + threadIdx.x;
    if (sample >= samples) return;
    // get the starting pointer for this sample
    real_type * theta_sample = theta + sample*parameters;

    // check each parameter
    for (int i=idx_begin; i<idx_end; ++i)
    {
        real_type u = (theta_sample[i]-low)/(high-low);
        theta_sample[i] = log(u/(1.0-u));
    }
}

// tophysical kernel (expit/sigmoid)
template <typename real_type>
__global__ void
_tophysical(real_type * const theta, const size_t samples, const size_t parameters,
        const size_t idx_begin, const size_t idx_end, const real_type low, const real_type high)
{
    // get the thread/sample id
    int sample = blockIdx.x*blockDim.x + threadIdx.x;
    if (sample >= samples) return;
    // get the starting pointer for this sample
    real_type * theta_sample = theta + sample*parameters;

    // check each parameter
    for (int i=idx_begin; i<idx_end; ++i)
    {
        theta_sample[i] = low+(high-low)/(1.0+exp(-theta_sample[i]));
    }
}
} // of namespace cudaUniformLogit_kernels

// end of file
