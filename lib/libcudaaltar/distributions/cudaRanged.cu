// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-2019 parasim inc
// (c) 2010-2019 california institute of technology
// all rights reserved
//
// Author(s): Hailiang Zhang, Lijun Zhu


// declarations 
#include "cudaRanged.h"
// cuda utilities
#include <pyre/cuda.h>
#include <curand_kernel.h>

// cuda kernel declarations
namespace cudaRanged_kernels {
    template<typename real_type> 
    __global__ void _verify(const real_type * const theta, int * const invalid, 
        const size_t samples, const size_t parameters, 
        const size_t idx_begin, const size_t idx_end, 
        const real_type low, const real_type high);
}

// verify whether samples are within range [low, high]
template<typename real_type> 
void altar::cuda::distributions::cudaRanged::
verify(const real_type * const theta, int * const invalid,
        const size_t samples, const size_t parameters,
        const size_t idx_begin, const size_t idx_end,
        const real_type low, const real_type high,
        const cudaStream_t stream)
{
    // determine the block/grid size
    // one thread for one sample
    dim3 blockSize (NTHREADS);
    dim3 gridSize (IDIVUP(samples, blockSize.x));
    // call cuda kernels
    cudaRanged_kernels::_verify<real_type><<<gridSize, blockSize, 0, stream>>>(
        theta, invalid, samples, parameters, idx_begin, idx_end, low, high);
    cudaCheckError("cudaRanged:: verify error");
}

// explicit instantiation
template void altar::cuda::distributions::cudaRanged::verify<float>(const float * const, int * const, const size_t, const size_t, 
                    const size_t, const size_t, const float, const float, cudaStream_t);
template void altar::cuda::distributions::cudaRanged::verify<double>(const double * const, int * const, const size_t, const size_t, 
                    const size_t, const size_t, const double, const double, cudaStream_t);
                    
//verify_kernel
template <typename real_type>
__global__ void
cudaRanged_kernels::
_verify(const real_type * const theta, int * const invalid, 
    const size_t samples, const size_t parameters, 
    const size_t idx_begin, const size_t idx_end, 
    const real_type low, const real_type high)
{
    int sample = blockIdx.x*blockDim.x + threadIdx.x;
    if (sample >= samples) return;
    
    // if already invalid, return
    if(invalid[sample]) return;
 
    // get the starting pointer for this sample
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

// end of file
