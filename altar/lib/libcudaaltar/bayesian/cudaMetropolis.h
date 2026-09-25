// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-2025 parasim inc
// (c) 2010-2025 california institute of technology
// all rights reserved
//
// Author(s): Hailiang Zhang, Lijun Zhu

// code guard
#ifndef altar_cuda_bayesian_cudaMetropolis_h
#define altar_cuda_bayesian_cudaMetropolis_h

#include <cuda_runtime.h>
// {matrix_view_t}/{vector_view_t}
#include "../support.h"

// place everything in the local namespace
namespace altar { namespace cuda {
    namespace bayesian {
        namespace cudaMetropolis {

            // compact the indices of the samples not flagged in {invalid} into the front of
            // {valid_sample_indices}, and leave the resulting count in {valid_samples[0]}
            // (a device scalar the caller reads back after synchronizing)
            void setValidSampleIndices(vector_view_t<int> valid_sample_indices,
                vector_view_t<int, true> invalid, vector_view_t<int> valid_samples,
                cudaStream_t stream=0);

            // gather the first {samples} valid proposals into {theta_candidate}:
            // theta_candidate[s, :] = theta_proposal[valid_sample_indices[s], :]
            template <typename realtype_t>
            void queueValidSamples(matrix_view_t<realtype_t, false> theta_candidate,
                matrix_view_t<realtype_t, true> theta_proposal,
                vector_view_t<int, true> valid_sample_indices,
                const size_t samples,
                cudaStream_t stream=0);

            // one Metropolis-Hastings accept/reject test per valid sample: on acceptance,
            // overwrite the corresponding row of {theta}/{prior}/{data}/{posterior} (indexed
            // by {valid_sample_indices[s]}) with the candidate's, and set
            // {acceptance_flag[s] = 1}
            template <typename realtype_t>
            void metropolisUpdate(matrix_view_t<realtype_t, false> theta,
                vector_view_t<realtype_t, false> prior,
                vector_view_t<realtype_t, false> data,
                vector_view_t<realtype_t, false> posterior,
                matrix_view_t<realtype_t, true> theta_candidate,
                vector_view_t<realtype_t, true> prior_candidate,
                vector_view_t<realtype_t, true> data_candidate,
                vector_view_t<realtype_t, true> posterior_candidate,
                vector_view_t<realtype_t, true> dices,
                vector_view_t<int> acceptance_flag,
                vector_view_t<int, true> valid_sample_indices,
                const size_t batch,
                cudaStream_t stream=0);

        } // of namespace cudaMetropolis
    } // of namespace bayesian
} }// of namespace altar::cuda


#endif //altar_cuda_bayesian_cudaMetropolis_h
// end of file
