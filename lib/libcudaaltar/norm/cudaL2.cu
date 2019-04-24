// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-2019 parasim inc
// (c) 2010-2019 california institute of technology
// all rights reserved
//
// Author(s): Hailiang Zhang, Lijun Zhu

// declarations 
#include "cudaL2.h"

// cuda utilities 
#include <pyre/cuda.h>

// cuda kernel declarations
namespace cudaL2_kernels {
    // norm ||x| 
    template <typename real_type>
    __global__ void _norm(const real_type * const data, real_type * const probability, 
        const size_t batch, const size_t parameters);
    // normllk constant-0.5 ||x||^2
    template <typename real_type>    
    __global__ void _normllk(const real_type * const data, real_type * const probability, 
        const size_t batch, const size_t parameters, const real_type constant);
}  
 
// l2 norm
// probability = ||data||
template <typename real_type>
void altar::cuda::norms::cudaL2::
norm(const real_type* const data, // input data , matrix(samples, parameters)
    real_type* const probability, // output norm, vector(samples)
    const size_t batch, // first batch of samples to be computed batch<=samples
    const size_t parameters, // number of parameters
    cudaStream_t stream)
{
    // determine the block/grid size
    // one thread for one 
    int blockSize = NTHREADS;
    int gridSize = IDIVUP(batch, blockSize);
    
    // call cuda kernels
    cudaL2_kernels::_norm<real_type><<<gridSize, blockSize, 0, stream>>>(
        data, probability, batch, parameters);
    cudaCheckError("cudaL2::L2norm error");
} 

// explicit specialization
template void altar::cuda::norms::cudaL2::norm<float>(
    const float * const, float * const, const size_t, const size_t, cudaStream_t);
template void altar::cuda::norms::cudaL2::norm<double>(
    const double * const, double * const, const size_t, const size_t, cudaStream_t);


// l2 normllk
// probability = -0.5 ||data||^2 + l2constant
template <typename real_type>
void altar::cuda::norms::cudaL2::
normllk(const real_type* const data, // input data , matrix(samples, parameters)
    real_type* const probability, // output norm, vector(samples)
    const size_t batch, // first batch of samples to be computed batch<=samples
    const size_t parameters, // number of parameters
    const real_type l2constant, // constant to be added to probability
    cudaStream_t stream)
{
    // determine the block/grid size
    // one thread for one 
    int blockSize = NTHREADS;
    int gridSize = IDIVUP(batch, blockSize);
    
    // call cuda kernels
    cudaL2_kernels::_normllk<real_type><<<gridSize, blockSize, 0, stream>>>(
        data, probability, batch, parameters, l2constant);
    cudaCheckError("cudaL2::L2normLLK error");
} 

// explicit specialization
template void altar::cuda::norms::cudaL2::normllk<float>(
    const float * const, float * const, const size_t, const size_t, const float, cudaStream_t);
template void altar::cuda::norms::cudaL2::normllk<double>(
    const double * const, double * const, const size_t, const size_t, const double, cudaStream_t);


// norm_kernel
template <typename real_type>
__global__ void
cudaL2_kernels::
_norm(const real_type* const data, real_type* const probability, 
    size_t batch, const size_t parameters)
{
    int sample = blockIdx.x*blockDim.x + threadIdx.x;
    if (sample >= batch) return;
    
    real_type prob = 0.0f;
    // get the pointer for a given sample
    const real_type* data_sample = data + sample*parameters;
    
    for(int i=0; i<parameters; ++i)
    {
        real_type value = data_sample[i];
        prob += value*value;
    }
    
    probability[sample] = sqrt(prob);
}

// normllk_kernel
template <typename real_type>
__global__ void
cudaL2_kernels::
_normllk(const real_type* const data, real_type* const probability, 
    size_t batch, const size_t parameters, 
    const real_type l2constant)
{
    int sample = blockIdx.x*blockDim.x + threadIdx.x;
    if (sample >= batch) return;
    
    real_type prob = 0.0f;
    // get the pointer for a given sample
    const real_type * data_sample = data + sample*parameters;
    
    for(int i=0; i<parameters; ++i)
    {
        real_type value = data_sample[i];
        prob += value*value;
    }

    probability[sample] = l2constant - 0.5*prob;
}

// end of file
