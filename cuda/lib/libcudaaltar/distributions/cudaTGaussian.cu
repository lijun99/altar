// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-2021 parasim inc
// (c) 2010-2021 california institute of technology
// all rights reserved
//
// Author(s):  Lijun Zhu


// declarations
#include "cudaTGaussian.h"
// cuda utilities
#include <pyre/cuda.h>
#include <curand_kernel.h>

// cuda kernel declarations
namespace cudaTGaussian_kernels {

    // wrap float and double rng routines into one template
    template<typename real_type>
    __device__ real_type
    _curandGenerateUniform(curandState_t *state);

    // sample
    template<typename real_type>
    __global__ void _sample(curandState_t * curand_states,
        real_type * const theta, const size_t samples, const size_t parameters,
        const size_t idx_begin, const size_t idx_end,
        const real_type mean, const real_type sigma,
        const real_type low, const real_type high);

    // log pdf
    template<typename real_type>
    __global__ void _logpdf(const real_type * const theta, real_type * const probability,
        const size_t samples, const size_t parameters,
        const size_t idx_begin, const size_t idx_end,
        const real_type mean, const real_type sigma,
        const real_type low, const real_type high);
}

// generate uniform random samples
template<typename real_type>
void altar::cuda::distributions::cudaTGaussian::
sample(real_type * const theta, const size_t samples, const size_t parameters,
                    const size_t idx_begin, const size_t idx_end,
                    const real_type mean, const real_type sigma,
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
    cudaTGaussian_kernels::_sample<real_type><<<gridSize, blockSize, 0, stream>>>(curand_states,
        theta, samples, parameters, idx_begin, idx_end, mean, sigma, low, high);
    cudaCheckError("cudaTGaussian::random generation error");

    cudaSafeCall(cudaFree(curand_states));
}

// explicit instantiation
template void altar::cuda::distributions::cudaTGaussian::sample<float>(float * const, const size_t, const size_t,
                    const size_t, const size_t, const float, const float, const float, const float, cudaStream_t);
template void altar::cuda::distributions::cudaTGaussian::sample<double>(double * const, const size_t, const size_t,
                    const size_t, const size_t, const double, const double, const double, const double, cudaStream_t);


// compute log probability
template <typename real_type>
void altar::cuda::distributions::cudaTGaussian::
logpdf(const real_type * const theta, real_type * const probability,
                    const size_t samples, const size_t parameters,
                    const size_t idx_begin, const size_t idx_end,
                    const real_type mean, const real_type sigma,
                    const real_type low, const real_type high,
                    cudaStream_t stream)
{
    int blockSize = NTHREADS;
    int gridSize = IDIVUP(samples, blockSize);

    // call cuda kernels
    cudaTGaussian_kernels::_logpdf<real_type><<<gridSize, blockSize, 0, stream>>>(
        theta, probability, samples, parameters, idx_begin, idx_end, mean, sigma, low, high);
    cudaCheckError("cudaTGaussian:: log_pdf error");
}

// explicit instantiation
template void altar::cuda::distributions::cudaTGaussian::logpdf<float>(const float * const, float * const, const size_t, const size_t,
                    const size_t, const size_t, const float, const float, const float, const float, cudaStream_t);
template void altar::cuda::distributions::cudaTGaussian::logpdf<double>(const double * const, double * const, const size_t, const size_t,
                    const size_t, const size_t, const double, const double, const double, const double, cudaStream_t);

// put explicit specialization in a namespace for compatibility with gcc6
namespace cudaTGaussian_kernels {

    template<>
    __device__ float
    _curandGenerateUniform<float>(curandState_t *state)
    {
        return curand_uniform(state);
    }

    template<>
    __device__ double
    _curandGenerateUniform<double>(curandState_t *state)
    {
        return curand_uniform_double(state);
    }
} // of namespace cudaTGaussian_kernels

// random sample generation
// note the support(low, high) are normalized Phi(x)
template <typename real_type>
__global__ void
cudaTGaussian_kernels::
_sample(curandState_t * curand_states,
    real_type * const theta, const size_t samples, const size_t parameters,
    const size_t idx_begin, const size_t idx_end,
    const real_type mean, const real_type sigma,
    const real_type low, const real_type high)
{
    // get the thread id as the sample
    int sample = blockIdx.x*blockDim.x + threadIdx.x;
    if (sample >= samples) return;

    // initialize seeds for each thread
    unsigned long long seed = (unsigned long long) clock64();
    curand_init(seed, sample, 0, &curand_states[sample]);

    // get the theta pointer for each sample
    real_type * theta_sample = theta + sample*parameters;

    real_type sqrt_two_sigma = sqrt(2.0)*sigma;
    // get the normalized support
    real_type range = high-low;

    // generate samples from idx_begin to idx_end
    for (int i=idx_begin; i<idx_end; ++i)
    {
        real_type temp = _curandGenerateUniform<real_type>(&curand_states[sample])*range + low;
        theta_sample[i] = erfinv(2.0*temp-1.0)*sqrt_two_sigma + mean;
    }
}

//log_pdf kernel
// note the support (low, high) are normalized Phi(x)
template <typename real_type>
__global__ void
cudaTGaussian_kernels::
_logpdf(const real_type * const theta, real_type * const probability, const size_t samples, const size_t parameters,
        const size_t idx_begin, const size_t idx_end, const real_type mean, const real_type sigma,
        const real_type low, const real_type high)
{
    // get the thread/sample id
    int sample = blockIdx.x*blockDim.x + threadIdx.x;
    if (sample >= samples) return;

    // get the starting pointer for this sample
    const real_type * theta_sample = theta + sample*parameters;

    // accumulated log_pdf for this sample
    real_type log_pdf = 0.0;

    // distribution constants
    real_type c1 = -log(sigma * sqrt(2.0*PI) * (high-low));
    real_type c2 = 0.5/(sigma*sigma);

    // iterate over each parameter in this sample
    for (int i=idx_begin; i<idx_end; ++i)
    {
        real_type mtmp = theta_sample[i]-mean;
        log_pdf += c1-mtmp*mtmp*c2;
    }
    // add to pdf of this sample
    probability[sample] += log_pdf;
}

// end of file
