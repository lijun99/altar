// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-2025 parasim inc
// (c) 2010-2025 california institute of technology
// all rights reserved
//
// Author(s):  Lijun Zhu

// code guard
#ifndef altar_cuda_distributions_cudaUniformLogit_h
#define altar_cuda_distributions_cudaUniformLogit_h

#include <cuda_runtime.h>

// place everything in the local namespace
namespace altar {
    namespace cuda {
        namespace distributions {
            namespace cudaUniformLogit {

                // sample and logpdf methods are defined in cudaLogistic

                // transform physical variables to sampling variables
                // logit bounded to unbounded
                template <typename real_type>
                void tosampling(real_type * const theta, const size_t samples, const size_t parameters,
                    const size_t idx_begin, const size_t idx_end,
                    const real_type low, const real_type high,
                    cudaStream_t stream=0);

                // transform sampling variables back to physical variables
                // expit unbounded to bounded
                template <typename real_type>
                void tophysical(real_type * const theta, const size_t samples, const size_t parameters,
                    const size_t idx_begin, const size_t idx_end,
                    const real_type low, const real_type high,
                    cudaStream_t stream=0);

            } // of namespace cudaUniformLogit
        } // of namespace distributions
    } // of namespace cuda
} // of namespace altar

#endif //altar_cuda_distributions_cudaUniformLogit_h
// end of file
