// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-2019 parasim inc
// (c) 2010-2019 california institute of technology
// all rights reserved
//
// Author(s): Hailiang Zhang, Lijun Zhu
// code guard
#ifndef altar_cuda_bayesian_cudaMetropolis_h
#define altar_cuda_bayesian_cudaMetropolis_h

#include <cuda_runtime.h>

// place everything in the local namespace
namespace altar { namespace cuda { 
    namespace bayesian {
        namespace cudaMetropolis {
            void setValidSampleIndices(int * const valid_sample_indices, const int * const invalid, 
                const int samples, int * valid_samples,
                cudaStream_t stream=0);
            template <typename realtype_t>
            void queueValidSamples(realtype_t * const theta_candidate, const realtype_t * const theta_proposal, 
                const int * const validSample_indices,
                const size_t samples, const size_t parameters, 
                cudaStream_t stream=0);
            template <typename realtype_t>
            void metropolisUpdate(realtype_t * const theta, realtype_t * const prior, 
                realtype_t * const data, realtype_t * const posterior,  
                const realtype_t * const theta_candidate, const realtype_t * const prior_candidate, 
                const realtype_t * const data_candidate, const realtype_t * const posterior_candidate,
                const realtype_t * const dices, int * const acceptance_flag, const int * const valid_sample_indices,
                const int samples, const int parameters,                 
                cudaStream_t stream=0);
        } // of namespace cudaMetropolis
    } // of namespace bayesian
} }// of namespace cualtar


#endif //altar_cuda_bayesian_cudaMetropolis_h
// end of file
