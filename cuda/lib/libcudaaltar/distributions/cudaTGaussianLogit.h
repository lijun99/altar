// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-2025 parasim inc
// (c) 2010-2025 california institute of technology
// all rights reserved
//
// Author(s):  Lijun Zhu

/// altar/cuda/distributions/cudaTGaussianLogit.h

// code guard
#ifndef altar_cuda_distributions_cudaTGaussianLogit_h
#define altar_cuda_distributions_cudaTGaussianLogit_h

#include <cuda_runtime.h>

// place everything in the local namespace
namespace altar {
    namespace cuda {
        namespace distributions {
            namespace cudaTGaussianLogit {

                // use cudaLogistic for sample and logpdf

                // transform physical variables to sampling variables
                // logit bounded to unbounded
                template <typename real_type>
                void tosampling(real_type * const theta, const size_t samples, const size_t parameters,
                    const size_t idx_begin, const size_t idx_end,
                    const real_type mean, const real_type sigma,
                    const real_type low, const real_type high,
                    cudaStream_t stream=0);

                // transform sampling variables back to physical variables
                // expit unbounded to bounded
                template <typename real_type>
                void tophysical(real_type * const theta, const size_t samples, const size_t parameters,
                    const size_t idx_begin, const size_t idx_end,
                    const real_type mean, const real_type sigma,
                    const real_type low, const real_type high,
                    cudaStream_t stream=0);
                
                
            } // of namespace cudaTGaussianLogit
        } // of namespace distributions
    } // of namespace cuda
} // of namespace altar

#endif //altar_cuda_distributions_cudaTGaussianLogit_h
// end of file
