// -*- C++ -*-
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//

#ifndef altar_cuda_distributions_cudaPolar_h
#define altar_cuda_distributions_cudaPolar_h

#include <cuda_runtime.h>
#include "../support.h"

namespace altar {
    namespace cuda {
        namespace distributions {
            namespace cudaPolar {

                // slips as a magnitude and a rake, in degrees, in the {count} columns from
                // {magnitude} and from {rake} of {theta}: {extended} is {theta} followed by the
                // components, magnitude cos(rake) then magnitude sin(rake)
                template <typename real_type>
                void extend(matrix_view_t<real_type> theta, matrix_view_t<real_type, false> extended,
                    const size_t magnitude, const size_t rake, const size_t count,
                    cudaStream_t stream=0);

                // the chain rule of {extend}: {gradient} = the gradient with respect to {theta}
                // of a function whose gradient with respect to {extended} is {gradient_extended}
                template <typename real_type>
                void fold(matrix_view_t<real_type> theta, matrix_view_t<real_type> gradient_extended,
                    matrix_view_t<real_type, false> gradient,
                    const size_t magnitude, const size_t rake, const size_t count,
                    cudaStream_t stream=0);

            } // of namespace cudaPolar
        } // of namespace distributions
    } // of namespace cuda
} // of namespace altar

#endif //altar_cuda_distributions_cudaPolar_h

// end of file
