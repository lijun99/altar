// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-2025 parasim inc
// (c) 2010-2025 california institute of technology
// all rights reserved
//
// Author(s): Lijun Zhu

//! cuda Leapfrog integrator for hamiltonian monte carlo 

// code guard
#ifndef altar_cuda_bayesian_cudaLeapfrog_h
#define altar_cuda_bayesian_cudaLeapfrog_h

#include <cuda_runtime.h>

// place everything in the local namespace
namespace altar { namespace cuda {
    namespace bayesian {
        namespace cudaLeapfrog {

            template <typename realtype_t>
            void sampleMomentum(
                realtype_t * const momentum,
                const size_t samples, const size_t parameters,
                cudaStream_t stream=0);

            template <typename realtype_t>
            void computePotentialAndGradient(
                const realtype_t * const prior,
                const realtype_t * const data,
                const realtype_t * const grad_prior,
                const realtype_t * const grad_data,
                realtype_t * const potential,
                realtype_t * const grad_potential,
                const realtype_t * const jacobian,
                const size_t samples, const size_t parameters,
                const realtype_t beta, const bool reparameterization,
                cudaStream_t stream=0);

            template <typename realtype_t>
            void kineticEnergy(
                const realtype_t * const momentum,
                realtype_t * const kinetic,
                const size_t samples, const size_t parameters,
                cudaStream_t stream=0);

            template <typename realtype_t>
            void updatePosition(
                realtype_t * const theta,
                const realtype_t * const momentum,
                const size_t samples, const size_t parameters,
                const realtype_t step, cudaStream_t stream=0);

            template <typename realtype_t>
            void updateMomentum(
                realtype_t * const momentum,
                const realtype_t * const gradU,
                const size_t samples, const size_t parameters,
                const realtype_t scale, cudaStream_t stream=0);

            template <typename realtype_t>
            void metropolis(
                const realtype_t * const deltaH,
                int * const mask,
                const size_t samples,
                cudaStream_t stream=0);

            template <typename realtype_t>
            void restoreRejected(
                realtype_t * const theta,
                const realtype_t * const theta_old,
                realtype_t * const momentum,
                const realtype_t * const momentum_old,
                const int * const mask,
                const size_t samples, const size_t parameters,
                cudaStream_t stream=0);

            template <typename realtype_t>
            void restoreMatrix(
                realtype_t * const current,
                const realtype_t * const backup,
                const int * const mask,
                const size_t samples, const size_t parameters,
                cudaStream_t stream=0);
        } // namespace cudaLeapfrog
    } // namespace bayesian
}} // namespace altar::cuda


#endif //altar_cuda_bayesian_cudaLeapfrog_h
// end of file
