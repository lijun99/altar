// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-2025 parasim inc
// (c) 2010-2025 california institute of technology
// all rights reserved
//
// Author(s): Hailiang Zhang, Lijun Zhu

// code guard
#ifndef altar_cuda_distributions_cudaRanged_h
#define altar_cuda_distributions_cudaRanged_h

#include <cuda_runtime.h>
// {matrix_view_t}/{vector_view_t}
#include "../support.h"

// place everything in the local namespace
namespace altar {
    namespace cuda {
        namespace distributions {
            namespace cudaRanged {

                // flag each sample whose parameters in [idx_begin, idx_end) fall outside
                // [low, high] by setting {invalid[sample] = 1}; a sample already flagged
                // (from an earlier distribution's own {verify} call) is left alone and not
                // re-checked
                template <typename real_type>
                void verify(matrix_view_t<real_type> theta, vector_view_t<int> invalid,
                    const size_t idx_begin, const size_t idx_end,
                    const real_type low, const real_type high,
                    cudaStream_t stream=0);

                // the per-parameter-bounds counterpart of {verify}: {low[j]}/{high[j]} apply
                // to parameter {idx_begin + j}
                template <typename real_type>
                void verify_unique(matrix_view_t<real_type> theta, vector_view_t<int> invalid,
                    const size_t idx_begin, const size_t idx_end,
                    vector_view_t<real_type, true> low, vector_view_t<real_type, true> high,
                    cudaStream_t stream=0);

                // clamp {theta[:, idx_begin:idx_end]} into [low, high], in place
                template <typename real_type>
                void constrain(matrix_view_t<real_type, false> theta,
                    const size_t idx_begin, const size_t idx_end,
                    const real_type low, const real_type high,
                    cudaStream_t stream=0);

            } // of namespace cudaRanged
        } // of namespace distributions
    } // of namespace cuda
} // of namespace altar

#endif //altar_cuda_distributions_cudaRanged_h
// end of file
