// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-2025 parasim inc
// (c) 2010-2025 california institute of technology
// all rights reserved
//
// Author(s): Hailiang Zhang, Lijun Zhu

// code guard
#ifndef altar_cuda_distributions_cudaUniform_h
#define altar_cuda_distributions_cudaUniform_h

#include <cuda_runtime.h>
// {matrix_view_t}/{vector_view_t}
#include "../support.h"

// place everything in the local namespace
namespace altar {
    namespace cuda {
        namespace distributions {
            namespace cudaUniform {

                // draw one sample per row of {theta}, for the parameters in [idx_begin,
                // idx_end), uniform over [low, high) -- the same bounds for every parameter
                // in range
                template <typename real_type>
                void sample(matrix_view_t<real_type, false> theta,
                    const size_t idx_begin, const size_t idx_end,
                    const real_type low, const real_type high,
                    cudaStream_t stream=0);

                // the same, but with per-parameter bounds: {low[j]}/{high[j]} apply to
                // parameter {idx_begin + j}
                template <typename real_type>
                void sample_unique(matrix_view_t<real_type, false> theta,
                    const size_t idx_begin, const size_t idx_end,
                    vector_view_t<real_type, true> low, vector_view_t<real_type, true> high,
                    cudaStream_t stream=0);

                // add each sample's log pdf, summed over [idx_begin, idx_end), into
                // {probability} (samples,) -- constant per sample, since a uniform's log pdf
                // doesn't depend on {theta} itself, only on the range
                template <typename real_type>
                void logpdf(matrix_view_t<real_type> theta, vector_view_t<real_type> probability,
                    const size_t idx_begin, const size_t idx_end,
                    const real_type low, const real_type high,
                    cudaStream_t stream=0);

                // the per-parameter-bounds counterpart of {logpdf}
                template <typename real_type>
                void logpdf_unique(matrix_view_t<real_type> theta, vector_view_t<real_type> probability,
                    const size_t idx_begin, const size_t idx_end,
                    vector_view_t<real_type, true> low, vector_view_t<real_type, true> high,
                    cudaStream_t stream=0);

            } // of namespace cudaUniform
        } // of namespace distributions
    } // of namespace cuda
} // of namespace altar

#endif //altar_cuda_distributions_cudaUniform_h
// end of file
