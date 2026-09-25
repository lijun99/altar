// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-2025 parasim inc
// (c) 2010-2025 california institute of technology
// all rights reserved
//
// Author(s): Lijun Zhu

// code guard
#ifndef altar_cuda_bayesian_cudaLangevin_h
#define altar_cuda_bayesian_cudaLangevin_h

#include <cuda_runtime.h>
// {matrix_view_t}/{vector_view_t}
#include "../support.h"

// place everything in the local namespace
namespace altar { namespace cuda {
    namespace bayesian {
        namespace cudaLangevin {

            // the SGLD update, one parameter at a time:
            // theta[:, index] += half_epsilon_t * (prior_gradient + datalikelihood_gradient) + eta_t
            template <typename realtype_t>
            void updateTheta(matrix_view_t<realtype_t, false> theta,
                vector_view_t<realtype_t, true> prior_gradient,
                vector_view_t<realtype_t, true> datalikelihood_gradient,
                const realtype_t half_epsilon_t, vector_view_t<realtype_t, true> eta_t,
                const size_t index,
                cudaStream_t stream=0);

            // the batched SGLD update, every parameter at once:
            // theta += half_epsilon_t * (alpha1*prior_gradient + alpha2*datalikelihood_gradient) + eta_t
            template <typename realtype_t>
            void updateThetaBatched(matrix_view_t<realtype_t, false> theta,
                const realtype_t alpha1, matrix_view_t<realtype_t, true> prior_gradient,
                const realtype_t alpha2, matrix_view_t<realtype_t, true> datalikelihood_gradient,
                const realtype_t half_epsilon_t, matrix_view_t<realtype_t, true> eta_t,
                cudaStream_t stream=0);

        } // of namespace cudaLangevin
    } // of namespace bayesian
} }// of namespace altar::cuda


#endif //altar_cuda_bayesian_cudaLangevin_h
// end of file
