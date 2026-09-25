// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//
// hailiang zhang
/// altar/cuda/distributions/cudaTGaussian.h
/// truncated gaussian distribution

// code guard
#ifndef altar_cuda_distributions_cudaTGaussian_h
#define altar_cuda_distributions_cudaTGaussian_h

#include <cuda_runtime.h>
// {matrix_view_t}/{vector_view_t}
#include "../support.h"

// place everything in the local namespace
namespace altar {
    namespace cuda {
        namespace distributions {
            namespace cudaTGaussian {

                // draw one sample per row of {theta}, for the parameters in [idx_begin,
                // idx_end); {low}/{high} are the *normalized* (Phi-space) support bounds
                template <typename real_type>
                void sample(matrix_view_t<real_type, false> theta,
                    const size_t idx_begin, const size_t idx_end,
                    const real_type mean, const real_type sigma,
                    const real_type low, const real_type high,
                    cudaStream_t stream=0);

                // add each sample's log pdf, summed over [idx_begin, idx_end), into
                // {probability} (samples,)
                template <typename real_type>
                void logpdf(matrix_view_t<real_type> theta, vector_view_t<real_type> probability,
                    const size_t idx_begin, const size_t idx_end,
                    const real_type mean, const real_type sigma,
                    const real_type low, const real_type high,
                    cudaStream_t stream=0);

            } // of namespace cudaTGaussian
        } // of namespace distributions
    } // of namespace cuda
} // of namespace altar

#endif //altar_cuda_distributions_cudaTGaussian_h
// end of file
