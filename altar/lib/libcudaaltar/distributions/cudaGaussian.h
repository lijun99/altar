// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//
// hailiang zhang
/// altar/cuda/distributions/cudaGaussian.h
/// Gaussian distribution

// code guard
#ifndef altar_cuda_distributions_cudaGaussian_h
#define altar_cuda_distributions_cudaGaussian_h

#include <cuda_runtime.h>
// {matrix_view_t}/{vector_view_t}
#include "../support.h"

// place everything in the local namespace
namespace altar {
    namespace cuda {
        namespace distributions {
            namespace cudaGaussian {

                // draw one sample per row of {theta}, for the parameters in [idx_begin,
                // idx_end); {theta} is (samples x parameters), written in place
                template <typename real_type>
                void sample(matrix_view_t<real_type, false> theta,
                    const size_t idx_begin, const size_t idx_end,
                    const real_type mean, const real_type sigma,
                    cudaStream_t stream=0);

                // add each sample's log pdf, summed over [idx_begin, idx_end), into
                // {probability} (samples,)
                template <typename real_type>
                void logpdf(matrix_view_t<real_type> theta, vector_view_t<real_type> probability,
                    const size_t idx_begin, const size_t idx_end,
                    const real_type mean, const real_type sigma,
                    cudaStream_t stream=0);

                // add the log pdf gradient with respect to parameter {index} into
                // {probability} (samples,); a no-op for samples where {index} falls outside
                // [idx_begin, idx_end)
                template <typename real_type>
                void logpdfgradient_i(matrix_view_t<real_type> theta, vector_view_t<real_type> probability,
                    const size_t idx_begin, const size_t idx_end, const size_t index,
                    const real_type mean, const real_type sigma,
                    cudaStream_t stream=0);

                // fill {probability} (samples x parameters) with the log pdf gradient with
                // respect to every parameter in [idx_begin, idx_end) (a plain assignment, not
                // an accumulation -- unlike {logpdfgradient_i} above)
                template <typename real_type>
                void logpdfgradient(matrix_view_t<real_type> theta, matrix_view_t<real_type, false> probability,
                    const size_t idx_begin, const size_t idx_end,
                    const real_type mean, const real_type sigma,
                    cudaStream_t stream=0);

            } // of namespace cudaGaussian
        } // of namespace distributions
    } // of namespace cuda
} // of namespace altar

#endif //altar_cuda_distributions_cudaGaussian_h
// end of file
