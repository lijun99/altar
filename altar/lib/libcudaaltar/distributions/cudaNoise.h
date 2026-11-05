// -*- C++ -*-
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//

#ifndef altar_cuda_distributions_cudaNoise_h
#define altar_cuda_distributions_cudaNoise_h

#include <cuda_runtime.h>
#include "../support.h"

namespace altar {
    namespace cuda {
        namespace distributions {
            namespace cudaNoise {

                // the data covariance of a model scaled by exp(2h), h in {column} of {theta}: the
                // first {batch} entries of {likelihood}, c - chi^2/2 with {normalization} c and
                // {observations} N, become c - N h - exp(-2h) chi^2/2
                template <typename real_type>
                void likelihood(matrix_view_t<real_type> theta, vector_view_t<real_type> likelihood,
                    const size_t column, const real_type observations, const real_type normalization,
                    const size_t batch, cudaStream_t stream=0);

                // and their gradients: the model's {data_gradient} times exp(-2h), and
                // -N + exp(-2h) chi^2, with chi^2 from the unscaled {likelihood}, added to
                // {column} of {gradient}
                template <typename real_type>
                void gradient(matrix_view_t<real_type> theta, vector_view_t<real_type, true> likelihood,
                    matrix_view_t<real_type, false> data_gradient, matrix_view_t<real_type, false> gradient,
                    const size_t column, const real_type observations, const real_type normalization,
                    const size_t batch, cudaStream_t stream=0);

            } // of namespace cudaNoise
        } // of namespace distributions
    } // of namespace cuda
} // of namespace altar

#endif //altar_cuda_distributions_cudaNoise_h

// end of file
