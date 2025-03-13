// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-2025 parasim inc
// (c) 2010-2025 california institute of technology
// all rights reserved
//
// Author(s): Lijun Zhu

// code guard
#ifndef altar_cuda_distributions_cudaLogistic_h
#define altar_cuda_distributions_cudaLogistic_h

#include <cuda_runtime.h>

// place everything in the local namespace
namespace altar {
    namespace cuda {
        namespace distributions {
            namespace cudaLogistic {
                // initialize random samples
                template <typename real_type>
                void sample(real_type * const theta,
                    const size_t samples, const size_t parameters,
                    const size_t idx_begin, const size_t idx_end,
                    cudaStream_t stream=0);

                // calculate log probability
                template <typename real_type>
                void logpdf(const real_type * const theta, real_type * const probability,
                    const size_t samples, const size_t parameters,
                    const size_t idx_begin, const size_t idx_end,
                    cudaStream_t stream=0);


            } // of namespace cudaLogistic
        } // of namespace distributions
    } // of namespace cuda
} // of namespace altar

#endif //altar_cuda_distributions_cudaLogistic_h
// end of file
