// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-2021 parasim inc
// (c) 2010-2021 california institute of technology
// all rights reserved
//
// Author(s): Lijun Zhu
// code guard
#ifndef altar_cuda_bayesian_cudaLangevin_h
#define altar_cuda_bayesian_cudaLangevin_h

#include <cuda_runtime.h>

// place everything in the local namespace
namespace altar { namespace cuda { 
    namespace bayesian {
        namespace cudaLangevin {
            // theta += (epsilon_t)/2 ( priorgradient + datalikelihood_gradient) + eta_t
            template <typename realtype_t>
            void updateTheta(realtype_t * const theta,
                const realtype_t * const prior_gradient,
                const realtype_t * const datalikelihood_gradient,
                const realtype_t half_epsilon_t, const realtype_t * const eta_t,
                const size_t samples, const size_t parameters, const size_t index,
                cudaStream_t stream=0);
        } // of namespace cudaLangevin
    } // of namespace bayesian
} }// of namespace cualtar


#endif //altar_cuda_bayesian_cudaLangevin_h
// end of file
