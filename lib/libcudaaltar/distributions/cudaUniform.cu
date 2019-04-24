// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-2019 parasim inc
// (c) 2010-2019 california institute of technology
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
    template<typename real_type> 
    __global__ void _sample(curandState_t * curand_states, 
        real_type * const theta, const size_t samples, const size_t parameters,  
        const size_t idx_begin, const size_t idx_end, 
        const real_type low, const real_type high);

    template<typename real_type> 
    __global__ void _verify(const real_type * const theta, int * const invalid, 
        const size_t samples, const size_t parameters, 
        const size_t idx_begin, const size_t idx_end, 
        const real_type low, const real_type high);

    template<typename real_type> 
    __global__ void _logpdf(const real_type * const theta, real_type * const probability,
        const size_t samples, const size_t parameters, 
        const size_t idx_begin, const size_t idx_end, 
        const real_type low, const real_type high);
}  
 
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

// explicit specialization
template void altar::cuda::distributions::cudaUniform::sample<float>(float * const, const size_t, const size_t, 
                    const size_t, const size_t, const float, const float, cudaStream_t);
template void altar::cuda::distributions::cudaUniform::sample<double>(double * const, const size_t, const size_t, 
                    const size_t, const size_t, const double, const double, cudaStream_t);


// verify whether samples are within range [low, high]
template<typename real_type> 
void altar::cuda::distributions::cudaUniform::
verify(const real_type * const theta, int * const invalid, 
                    const size_t samples, const size_t parameters, 
                    const size_t idx_begin, const size_t idx_end,
                    const real_type low, const real_type high,
                    const cudaStream_t stream)
{
    // determine the block/grid size
    // one thread for one sample
    int blockSize = NTHREADS;
    int gridSize = IDIVUP(samples, blockSize);
    // call cuda kernels
    cudaUniform_kernels::_verify<real_type><<<gridSize, blockSize, 0, stream>>>(
        theta, invalid, samples, parameters, idx_begin, idx_end, low, high);
    cudaCheckError("cudaUniform:: verify error");
}

// explicit specialization
template void altar::cuda::distributions::cudaUniform::verify<float>(const float * const, int * const, const size_t, const size_t, 
                    const size_t, const size_t, const float, const float, cudaStream_t);
template void altar::cuda::distributions::cudaUniform::verify<double>(const double * const, int * const, const size_t, const size_t, 
                    const size_t, const size_t, const double, const double, cudaStream_t);
                    
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

// explicit specialization
template void altar::cuda::distributions::cudaUniform::logpdf<float>(const float * const, float * const, const size_t, const size_t, 
                    const size_t, const size_t, const float, const float, cudaStream_t);
template void altar::cuda::distributions::cudaUniform::logpdf<double>(const double * const, double * const, const size_t, const size_t, 
                    const size_t, const size_t, const double, const double, cudaStream_t);
 
//random_generation_kernel 
// double precision version
template <>
__global__ void
cudaUniform_kernels::
_sample<double>(curandState_t * curand_states, 
    double * const theta, const size_t samples, const size_t parameters, 
    const size_t idx_begin, const size_t idx_end, 
    const double low, const double high)
{
    int sample = blockIdx.x*blockDim.x + threadIdx.x;
    if (sample >= samples) return;
    
    // initialize seeds for each thread    
    unsigned long long seed = (unsigned long long) clock64(); 
    curand_init(seed, sample, 0, &curand_states[sample]); 
    
    double range = high-low;
    // get the theta pointer for each sample 
    double * theta_sample = theta + sample*parameters;
    
    // generate samples from idx_begin to idx_end 
    for (int i=idx_begin; i<idx_end; ++i)
    {
        theta_sample[i] = curand_uniform_double(&curand_states[sample])*range + low;
    }
}
//single precision version
template <>
__global__ void
cudaUniform_kernels::
_sample<float>(curandState_t * curand_states, 
    float * const theta, const size_t samples, const size_t parameters, 
    const size_t idx_begin, const size_t idx_end, 
    const float low, const float high)
{
    int sample = blockIdx.x*blockDim.x + threadIdx.x;
    if (sample >= samples) return;
    
    // initialize seeds for each thread    
    unsigned long long seed = (unsigned long long) clock64(); 
    curand_init(seed, sample, 0, &curand_states[sample]); 
    
    float range = high-low;
    // get the theta pointer for each sample 
    float * theta_sample = theta + sample*parameters;
    
    // generate samples from idx_begin to idx_end 
    for (int i=idx_begin; i<idx_end; ++i)
    {
        theta_sample[i] = curand_uniform(&curand_states[sample])*range + low;
    }
}

//verify_kernel 
template <typename real_type>
__global__ void
cudaUniform_kernels::
_verify(const real_type * const theta, int * const invalid, 
    const size_t samples, const size_t parameters, 
    const size_t idx_begin, const size_t idx_end, 
    const real_type low, const real_type high)
{
    int sample = blockIdx.x*blockDim.x + threadIdx.x;
    if (sample >= samples) return;
    
    // if already invalid, return
    if(invalid[sample]) return;
 
    // get 
    const real_type * theta_sample = theta + sample*parameters;
    
    // check each parameter
    for (int i=idx_begin; i<idx_end; ++i)
    {
        real_type value = theta_sample[i];
        if(value < low || value > high) {
            invalid[sample] = 1;
            return;
        }
    }
}

//log_pdf kernel
template <typename real_type>
__global__ void
cudaUniform_kernels::
_logpdf(const real_type * const theta, real_type * const probability, const size_t samples, const size_t parameters, 
        const size_t idx_begin, const size_t idx_end, const real_type low, const real_type high)
{
    // get the thread/sample id
    int sample = blockIdx.x*blockDim.x + threadIdx.x;
    if (sample >= samples) return;

    real_type  log_pdf = log(high-low)*(idx_end-idx_begin);
    probability[sample] += log_pdf;
}


// end of file
