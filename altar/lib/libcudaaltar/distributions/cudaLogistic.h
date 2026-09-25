// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//

// code guard
#ifndef altar_cuda_distributions_cudaLogistic_h
#define altar_cuda_distributions_cudaLogistic_h

#include <cuda_runtime.h>
// {matrix_view_t}/{vector_view_t}
#include "../support.h"

// place everything in the local namespace
namespace altar {
    namespace cuda {
        namespace distributions {
            namespace cudaLogistic {

                // draw one sample per row of {theta}, for the parameters in [idx_begin,
                // idx_end), from the standard logistic distribution
                template <typename real_type>
                void sample(matrix_view_t<real_type, false> theta,
                    const size_t idx_begin, const size_t idx_end,
                    cudaStream_t stream=0);

                // add each sample's log pdf, summed over [idx_begin, idx_end), into
                // {probability} (samples,)
                template <typename real_type>
                void logpdf(matrix_view_t<real_type> theta, vector_view_t<real_type> probability,
                    const size_t idx_begin, const size_t idx_end,
                    cudaStream_t stream=0);

                // fill {probability} (samples x parameters) with the log pdf gradient with
                // respect to every parameter in [idx_begin, idx_end) (a plain assignment, not
                // an accumulation)
                template <typename real_type>
                void logpdfgradient(matrix_view_t<real_type> theta, matrix_view_t<real_type, false> probability,
                    const size_t idx_begin, const size_t idx_end,
                    cudaStream_t stream=0);

            } // of namespace cudaLogistic
        } // of namespace distributions
    } // of namespace cuda
} // of namespace altar

#endif //altar_cuda_distributions_cudaLogistic_h
// end of file
