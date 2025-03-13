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
#include <pyre/cuda.h>
#include <curand_kernel.h>

// cuda kernel declarations
namespace cudaLogistic_kernels {
    template<typename real_type>
    __global__ void _sample(curandState_t * curand_states,
        real_type * const theta, const size_t samples, const size_t parameters,
        const size_t idx_begin, const size_t idx_end);

    template<typename real_type>
    __global__ void _logpdf(const real_type * const theta, real_type * const probability,
        const size_t samples, const size_t parameters,
        const size_t idx_begin, const size_t idx_end);

} // of namespace cudaLogistic_kernels

// generate uniform random samples
template<typename real_type>
void altar::cuda::distributions::cudaLogistic::
sample(real_type * const theta, const size_t samples, const size_t parameters,
                    const size_t idx_begin, const size_t idx_end,
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
    cudaLogistic_kernels::_sample<real_type><<<gridSize, blockSize, 0, stream>>>(curand_states,
        theta, samples, parameters, idx_begin, idx_end);
    cudaCheckError("cudaLogistic::random generation error");

    cudaSafeCall(cudaFree(curand_states));
}

// explicit instantiation
template void altar::cuda::distributions::cudaLogistic::sample<float>(float * const, const size_t, const size_t,
                    const size_t, const size_t, cudaStream_t);
template void altar::cuda::distributions::cudaLogistic::sample<double>(double * const, const size_t, const size_t,
                    const size_t, const size_t, cudaStream_t);

// compute log probability
template <typename real_type>
void altar::cuda::distributions::cudaLogistic::
logpdf(const real_type * const theta, real_type * const probability,
                    const size_t samples, const size_t parameters,
                    const size_t idx_begin, const size_t idx_end,
                    cudaStream_t stream)
{
    int blockSize = NTHREADS;
    int gridSize = IDIVUP(samples, blockSize);
    // call cuda kernels
    cudaLogistic_kernels::_logpdf<real_type><<<gridSize, blockSize, 0, stream>>>(
        theta, probability, samples, parameters, idx_begin, idx_end);
    cudaCheckError("cudaLogistic:: log_pdf error");
}

// explicit instantiation
template void altar::cuda::distributions::cudaLogistic::logpdf<float>(const float * const, float * const, const size_t, const size_t,
                    const size_t, const size_t, cudaStream_t);
template void altar::cuda::distributions::cudaLogistic::logpdf<double>(const double * const, double * const, const size_t, const size_t,
                    const size_t, const size_t, cudaStream_t);

namespace cudaLogistic_kernels {

//random_generation_kernel
template <typename real_type>
__global__ void
_sample(curandState_t * curand_states,
    real_type * const theta, const size_t samples, const size_t parameters,
    const size_t idx_begin, const size_t idx_end)
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
        const size_t idx_begin, const size_t idx_end)
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

} // of namespace cudaLogistic_kernels

// end of file
