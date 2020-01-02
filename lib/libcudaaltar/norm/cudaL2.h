// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-2020 parasim inc
// (c) 2010-2020 california institute of technology
// all rights reserved
//
// Author(s): Hailiang Zhang, Lijun Zhu

// code guard
#ifndef altar_cuda_norms_cudaL2_h
#define altar_cuda_norms_cudaL2_h

#include <cuda_runtime.h>

// place everything in the local namespace
namespace altar {
    namespace cuda {
        namespace norms {
            namespace cudaL2 {
                
                // compute l2 norm for a batch of data (samples x parameters)
                // return probability (samples)
                // probability = ||data||
                template <typename real_type>
                void norm(const real_type * const data, real_type * const probability, 
                    const size_t batch, const size_t parameters, cudaStream_t stream=0);

                // compute l2 llk directly for a batch of data (samples x parameters)
                // return probability (samples)
                // probability = constant - 0.5 ||data||^2
                template <typename real_type>
                void normllk(const real_type * const data, real_type * const probability, 
                    const size_t batch, const size_t parameters,
                    const real_type constant=0.0, cudaStream_t stream=0);
                    
            } // of namespace cudaL2
        } // of namespace norms
    } // of namespace cuda
} // of namespace altar

#endif //altar_cuda_norms_cudaL2_h
// end of file
