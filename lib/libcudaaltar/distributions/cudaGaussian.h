// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-2019 parasim inc
// (c) 2010-2019 california institute of technology
// all rights reserved
//
// Author(s): Hailiang Zhang, Lijun Zhu

/// altar/cuda/distributions/cudaGaussian.h
/// Uniform Distribution

// code guard
#ifndef altar_cuda_distributions_cudaGaussian_h
#define altar_cuda_distributions_cudaGaussian_h

#include <cuda_runtime.h>

// place everything in the local namespace
namespace altar {
    namespace cuda {
        namespace distributions {
            namespace cudaGaussian {
                // initialize random samples
                template <typename real_type>
                void sample(real_type * const theta, const size_t samples, const size_t parameters, 
                    const size_t idx_begin, const size_t idx_end,
                    const real_type mean, const real_type sigma,
                    cudaStream_t stream=0);
                
                // calculate log probability
                template <typename real_type>
                void logpdf(const real_type * const theta, real_type * const probability, 
                    const size_t samples, const size_t parameters, 
                    const size_t idx_begin, const size_t idx_end,
                    const real_type mean, const real_type sigma,
                    cudaStream_t stream=0);

            } // of namespace cudaGaussian
        } // of namespace distributions
    } // of namespace cuda
} // of namespace altar

#endif //altar_cuda_distributions_cudaGaussian_h
// end of file
