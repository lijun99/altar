// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-2021 parasim inc
// (c) 2010-2021 california institute of technology
// all rights reserved
//
// Author(s): Hailiang Zhang, Lijun Zhu


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
    __global__ void _sample(curandState_t * curand_states,
        real_type * const theta, const size_t samples, const size_t parameters,
        const size_t idx_begin, const size_t idx_end,
        const real_type low, const real_type high);

    template<typename real_type>
    __global__ void _logpdf(const real_type * const theta, real_type * const probability,
        const size_t samples, const size_t parameters,
        const size_t idx_begin, const size_t idx_end,
        const real_type low, const real_type high);

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

// generate uniform random samples
template<typename real_type>
void altar::cuda::distributions::cudaUniformLogit::
sample(real_type * const theta, const size_t samples, const size_t parameters,
                    const size_t idx_begin, const size_t idx_end,
                    const real_type low, const real_type high,
                    cudaStream_t stream)
{
    // determine the block/grid size
    // one thread for one sample
    int blockSize = NTHREADS;
    int gridSize = IDIVUP(samples, blockSize);

    // allocate
    curandState_t *curand_states;
    cudaSafeCall(cudaMalloc((void**)&curand_states, blockSize*gridSize*sizeof(curandState)));

    // call cuda kernels
    cudaUniformLogit_kernels::_sample<real_type><<<gridSize, blockSize, 0, stream>>>(curand_states,
        theta, samples, parameters, idx_begin, idx_end, low, high);
    cudaCheckError("cudaUniformLogit::random generation error");

    cudaSafeCall(cudaFree(curand_states));
}

// explicit instantiation
template void altar::cuda::distributions::cudaUniformLogit::sample<float>(float * const, const size_t, const size_t,
                    const size_t, const size_t, const float, const float, cudaStream_t);
template void altar::cuda::distributions::cudaUniformLogit::sample<double>(double * const, const size_t, const size_t,
                    const size_t, const size_t, const double, const double, cudaStream_t);

// compute log probability
template <typename real_type>
void altar::cuda::distributions::cudaUniformLogit::
logpdf(const real_type * const theta, real_type * const probability,
                    const size_t samples, const size_t parameters,
                    const size_t idx_begin, const size_t idx_end,
                    const real_type low, const real_type high,
                    cudaStream_t stream)
{
    int blockSize = NTHREADS;
    int gridSize = IDIVUP(samples, blockSize);
    // call cuda kernels
    cudaUniformLogit_kernels::_logpdf<real_type><<<gridSize, blockSize, 0, stream>>>(
        theta, probability, samples, parameters, idx_begin, idx_end, low, high);
    cudaCheckError("cudaUniformLogit:: log_pdf error");
}

// explicit instantiation
template void altar::cuda::distributions::cudaUniformLogit::logpdf<float>(const float * const, float * const, const size_t, const size_t,
                    const size_t, const size_t, const float, const float, cudaStream_t);
template void altar::cuda::distributions::cudaUniformLogit::logpdf<double>(const double * const, double * const, const size_t, const size_t,
                    const size_t, const size_t, const double, const double, cudaStream_t);


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

template <typename real_type>
__global__ void
_sample(curandState_t * curand_states,
    real_type * const theta, const size_t samples, const size_t parameters,
    const size_t idx_begin, const size_t idx_end,
    const real_type low, const real_type high)
{
    int sample = blockIdx.x*blockDim.x + threadIdx.x;
    if (sample >= samples) return;

    // initialize seeds for each thread
    unsigned long long seed = (unsigned long long) clock64();
    curand_init(seed, sample, 0, &curand_states[sample]);

    // get the theta pointer for each sample
    real_type * theta_sample = theta + sample*parameters;

    // generate samples from idx_begin to idx_end
    for (int i=idx_begin; i<idx_end; ++i)
    {
        // generate a uniform random number [0, 1]
        real_type ran_num = altar::cuda::distributions::curandUniform<real_type>(&curand_states[sample]);
        // convert it to a logit variable
        theta_sample[i] = log(ran_num/(1.0-ran_num));
    }
}

//log_pdf kernel
template <typename real_type>
__global__ void
_logpdf(const real_type * const theta, real_type * const probability, const size_t samples, const size_t parameters,
        const size_t idx_begin, const size_t idx_end, const real_type low, const real_type high)
{
    // get the thread/sample id
    int sample = blockIdx.x*blockDim.x + threadIdx.x;
    if (sample >= samples) return;

    // get the starting pointer for this sample
    const real_type * theta_sample = theta + sample*parameters;

    // accumulated log_pdf for this sample
    real_type log_pdf = 0.0;

    // iterate over parameters in this dataset and in this sample
    for (int i=idx_begin; i<idx_end; ++i)
    {
        real_type theta_i = theta_sample[i];
        // logistic function pdf e^x/(1+e^x)^2
        log_pdf += theta_i - 2.0*log(1.0+exp(theta_i));
    }

    probability[sample] += log_pdf;
}

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
