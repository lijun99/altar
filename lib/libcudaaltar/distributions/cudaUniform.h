// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-2019 parasim inc
// (c) 2010-2019 california institute of technology
// all rights reserved
//
// Author(s): Hailiang Zhang, Lijun Zhu

// code guard
#ifndef altar_cuda_distributions_cudaUniform_h
#define altar_cuda_distributions_cudaUniform_h

#include <cuda_runtime.h>

// place everything in the local namespace
namespace altar {
    namespace cuda {
        namespace distributions {
            namespace cudaUniform {
                // initialize random samples
                template <typename real_type>
                void sample(real_type * const theta, const size_t samples, const size_t parameters, 
                    const size_t idx_begin, const size_t idx_end,
                    const real_type low, const real_type high,
                    cudaStream_t stream=0);
                
                // verify the validity of samples
                template <typename real_type>
                void verify(const real_type * const theta, int * const invalid, 
                    const size_t samples, const size_t parameters, 
                    const size_t idx_begin, const size_t idx_end,
                    const real_type low, const real_type high,
                    cudaStream_t stream=0);
                
                // calculate log probability
                template <typename real_type>
                void logpdf(const real_type * const theta, real_type * const probability, 
                    const size_t samples, const size_t parameters, 
                    const size_t idx_begin, const size_t idx_end,
                    const real_type low, const real_type high,
                    cudaStream_t stream=0);
                
                
            } // of namespace cudaUniform
        } // of namespace distributions
    } // of namespace cuda
} // of namespace altar

#endif //altar_cuda_distributions_cudaUniform_h
// end of file
