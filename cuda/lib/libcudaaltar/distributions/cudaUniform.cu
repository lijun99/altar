// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-2021 parasim inc
// (c) 2010-2021 california institute of technology
// all rights reserved
//
// Author(s): Hailiang Zhang, Lijun Zhu


// declarations
#include "cudaUniform.h"
// cuda utilities
#include <pyre/cuda.h>
#include <curand_kernel.h>

// cuda kernel declarations
namespace cudaUniform_kernels {

    // use template to call curand_uniform
    template <typename real_type>
    __inline__ __device__
    real_type cuRand_Uniform(curandState_t * curand_state);

    template <>
    __inline__ __device__
    float cuRand_Uniform<float>(curandState_t * curand_state)
    {
        return curand_uniform(curand_state);
    }

    template <>
    __inline__ __device__
    double cuRand_Uniform<double>(curandState_t * curand_state)
    {
        return curand_uniform_double(curand_state);
    }

    // sample
    template<typename real_type>
    __global__ void _sample(curandState_t * curand_states,
        real_type * const theta, const size_t samples, const size_t parameters,
        const size_t idx_begin, const size_t idx_end,
        const real_type low, const real_type high);

    template<typename real_type>
    __global__ void _sample_unique(curandState_t * curand_states,
        real_type * const theta, const size_t samples, const size_t parameters,
        const size_t idx_begin, const size_t idx_end,
        const real_type * const low, const real_type * const high);

    template<typename real_type>
    __global__ void _logpdf(const real_type * const theta, real_type * const probability,
        const size_t samples, const size_t parameters,
        const size_t idx_begin, const size_t idx_end,
        const real_type low, const real_type high);

    template<typename real_type>
    __global__ void _logpdf_unique(const real_type * const theta, real_type * const probability,
        const size_t samples, const size_t parameters,
        const size_t idx_begin, const size_t idx_end,
        const real_type * const low, const real_type * const high);

} // of namespace cudaUniform_kernels

// generate uniform random samples
template<typename real_type>
void altar::cuda::distributions::cudaUniform::
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
    cudaUniform_kernels::_sample<real_type><<<gridSize, blockSize, 0, stream>>>(curand_states,
        theta, samples, parameters, idx_begin, idx_end, low, high);
    cudaCheckError("cudaUniform::random generation error");

    cudaSafeCall(cudaFree(curand_states));
}

// explicit instantiation
template void altar::cuda::distributions::cudaUniform::sample<float>(float * const, const size_t, const size_t,
                    const size_t, const size_t, const float, const float, cudaStream_t);
template void altar::cuda::distributions::cudaUniform::sample<double>(double * const, const size_t, const size_t,
                    const size_t, const size_t, const double, const double, cudaStream_t);

// generate uniform random samples
template<typename real_type>
void altar::cuda::distributions::cudaUniform::
sample_unique(real_type * const theta, const size_t samples, const size_t parameters,
                    const size_t idx_begin, const size_t idx_end,
                    const real_type * const low, const real_type * const high,
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
    cudaUniform_kernels::_sample_unique<real_type><<<gridSize, blockSize, 0, stream>>>(curand_states,
        theta, samples, parameters, idx_begin, idx_end, low, high);
    cudaCheckError("cudaUniform::random unique generation error");

    cudaSafeCall(cudaFree(curand_states));
}

// explicit instantiation
template void altar::cuda::distributions::cudaUniform::sample_unique<float>(float * const, const size_t, const size_t,
                    const size_t, const size_t, const float * const, const float * const, cudaStream_t);
template void altar::cuda::distributions::cudaUniform::sample_unique<double>(double * const, const size_t, const size_t,
                    const size_t, const size_t, const double * const, const double * const, cudaStream_t);

// compute log probability
template <typename real_type>
void altar::cuda::distributions::cudaUniform::
logpdf(const real_type * const theta, real_type * const probability,
                    const size_t samples, const size_t parameters,
                    const size_t idx_begin, const size_t idx_end,
                    const real_type low, const real_type high,
                    cudaStream_t stream)
{
    int blockSize = NTHREADS;
    int gridSize = IDIVUP(samples, blockSize);
    // call cuda kernels
    cudaUniform_kernels::_logpdf<real_type><<<gridSize, blockSize, 0, stream>>>(
        theta, probability, samples, parameters, idx_begin, idx_end, low, high);
    cudaCheckError("cudaUniform:: log_pdf error");
}

// explicit instantiation
template void altar::cuda::distributions::cudaUniform::logpdf<float>(const float * const, float * const, const size_t, const size_t,
                    const size_t, const size_t, const float, const float, cudaStream_t);
template void altar::cuda::distributions::cudaUniform::logpdf<double>(const double * const, double * const, const size_t, const size_t,
                    const size_t, const size_t, const double, const double, cudaStream_t);


// compute log probability
template <typename real_type>
void altar::cuda::distributions::cudaUniform::
logpdf_unique(const real_type * const theta, real_type * const probability,
                    const size_t samples, const size_t parameters,
                    const size_t idx_begin, const size_t idx_end,
                    const real_type * const low, const real_type * const high,
                    cudaStream_t stream)
{
    int blockSize = NTHREADS;
    int gridSize = IDIVUP(samples, blockSize);
    // call cuda kernels
    cudaUniform_kernels::_logpdf_unique<real_type><<<gridSize, blockSize, 0, stream>>>(
        theta, probability, samples, parameters, idx_begin, idx_end, low, high);
    cudaCheckError("cudaUniform:: log_pdf_unique error");
}

// explicit instantiation
template void altar::cuda::distributions::cudaUniform::logpdf_unique<float>(const float * const, float * const, const size_t, const size_t,
                    const size_t, const size_t, const float * const, const float * const, cudaStream_t);
template void altar::cuda::distributions::cudaUniform::logpdf_unique<double>(const double * const, double * const, const size_t, const size_t,
                    const size_t, const size_t, const double * const, const double * const, cudaStream_t);


namespace cudaUniform_kernels {

//random_generation_kernel
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

    real_type range = high-low;
    // get the theta pointer for each sample
    real_type * theta_sample = theta + sample*parameters;

    // generate samples from idx_begin to idx_end
    for (int i=idx_begin; i<idx_end; ++i)
    {
        theta_sample[i] = cuRand_Uniform<real_type>(&curand_states[sample])*range + low;
    }
}

//random_generation_kernel
template <typename real_type>
__global__ void
_sample_unique(curandState_t * curand_states,
    real_type * const theta, const size_t samples, const size_t parameters,
    const size_t idx_begin, const size_t idx_end,
    const real_type * const low, const real_type * const high)
{
    int sample = blockIdx.x*blockDim.x + threadIdx.x;
    if (sample >= samples) return;

    // initialize seeds for each thread
    unsigned long long seed = (unsigned long long) clock64();
    curand_init(seed, sample, 0, &curand_states[sample]);

    // get the theta pointer for each sample
    real_type * theta_sample = theta + sample*parameters;

    // generate samples from idx_begin to idx_end
    for (int i=idx_begin, j=0; i<idx_end; ++i, ++j)
    {
        real_type range = high[j]-low[j];
        theta_sample[i] = cuRand_Uniform<real_type>(&curand_states[sample])*range + low[j];
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
    //  log(1/(high-low))*number of  parameters
    // note size_t is always non-negative
    real_type  log_pdf = -log(high-low)*(idx_end-idx_begin);
    probability[sample] += log_pdf;
}

//log_pdf kernel
template <typename real_type>
__global__ void
_logpdf_unique(const real_type * const theta, real_type * const probability, const size_t samples, const size_t parameters,
        const size_t idx_begin, const size_t idx_end, const real_type * const low, const real_type * const high)
{
    // get the thread/sample id
    int sample = blockIdx.x*blockDim.x + threadIdx.x;
    if (sample >= samples) return;
    //  log(1/(high-low))*number of  parameters

    // note size_t is always non-negative
    for (int i=idx_begin, j=0; i<idx_end; ++i, ++j)
    {
        real_type log_pdf = -log(high[j]-low[j]);
        probability[sample] += log_pdf;
    }
}

} // of namespace cudaUniform_kernels

// end of file
