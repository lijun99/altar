// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//

// code guard
#ifndef altar_cuda_distributions_cudaLogitTransform_h
#define altar_cuda_distributions_cudaLogitTransform_h

#include <cuda_runtime.h>
// {matrix_view_t}/{vector_view_t}
#include "../support.h"

// place everything in the local namespace
namespace altar {
    namespace cuda {
        namespace distributions {
            namespace cudaLogitTransform {

                // theta[:, idx_begin:idx_end] <- low + (high-low)*sigmoid(theta), in place;
                // maps sampling space to physical space
                template <typename real_type>
                void to_physical(matrix_view_t<real_type, false> theta,
                    const size_t idx_begin, const size_t idx_end,
                    const real_type low, const real_type high,
                    cudaStream_t stream=0);

                // theta[:, idx_begin:idx_end] <- logit((theta-low)/(high-low)), in place; the
                // inverse of {to_physical}
                template <typename real_type>
                void to_sampling(matrix_view_t<real_type, false> theta,
                    const size_t idx_begin, const size_t idx_end,
                    const real_type low, const real_type high,
                    cudaStream_t stream=0);

                // fill {jacobian[:, idx_begin:idx_end]} with d(physical)/d(sampling) =
                // (high-low)*sig*(1-sig), sig=(theta-low)/(high-low); {theta} is PHYSICAL
                // space (read-only), {jacobian} a separate output buffer
                template <typename real_type>
                void jacobian(matrix_view_t<real_type> theta, matrix_view_t<real_type, false> jacobian,
                    const size_t idx_begin, const size_t idx_end,
                    const real_type low, const real_type high,
                    cudaStream_t stream=0);

                // fill {gradient[:, idx_begin:idx_end]} with d/d(sampling)[log(sig) +
                // log(1-sig)] = 1 - 2*sig; {theta} is PHYSICAL space (read-only)
                template <typename real_type>
                void jacobian_gradient(matrix_view_t<real_type> theta, matrix_view_t<real_type, false> gradient,
                    const size_t idx_begin, const size_t idx_end,
                    const real_type low, const real_type high,
                    cudaStream_t stream=0);

                // add each sample's standard-logistic log-pdf, summed over
                // [idx_begin, idx_end), into {likelihood} (samples,) -- an accumulation, like
                // {cudaUniform::logpdf}; {theta} is PHYSICAL space (read-only)
                template <typename real_type>
                void log_jacobian(matrix_view_t<real_type> theta, vector_view_t<real_type> likelihood,
                    const size_t idx_begin, const size_t idx_end,
                    const real_type low, const real_type high,
                    cudaStream_t stream=0);

                // gradient[:, idx_begin:idx_end] <- gradient*(high-low)*sig*(1-sig) +
                // (1 - 2*sig), in place: a physical-space prior gradient turned into the
                // sampling-space one; {theta} is PHYSICAL space (read-only)
                template <typename real_type>
                void chain_gradient(matrix_view_t<real_type> theta, matrix_view_t<real_type, false> gradient,
                    const size_t idx_begin, const size_t idx_end,
                    const real_type low, const real_type high,
                    cudaStream_t stream=0);

            } // of namespace cudaLogitTransform
        } // of namespace distributions
    } // of namespace cuda
} // of namespace altar

#endif //altar_cuda_distributions_cudaLogitTransform_h
// end of file
