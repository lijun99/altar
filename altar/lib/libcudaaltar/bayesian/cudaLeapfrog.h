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
// {matrix_view_t}/{vector_view_t}
#include "../support.h"

// place everything in the local namespace
namespace altar { namespace cuda {
    namespace bayesian {
        namespace cudaLeapfrog {

            // fill {momentum} with iid draws from N(0, 1)
            template <typename realtype_t>
            void sampleMomentum(
                matrix_view_t<realtype_t, false> momentum,
                cudaStream_t stream=0);

            // potential[s] = -(prior[s] + beta * data[s]); grad_potential[s, :] =
            // -(grad_prior[s, :] + beta * grad_data[s, :])
            template <typename realtype_t>
            void computePotentialAndGradient(
                vector_view_t<realtype_t, true> prior,
                vector_view_t<realtype_t, true> data,
                matrix_view_t<realtype_t, true> grad_prior,
                matrix_view_t<realtype_t, true> grad_data,
                vector_view_t<realtype_t, false> potential,
                matrix_view_t<realtype_t, false> grad_potential,
                const realtype_t beta,
                cudaStream_t stream=0);

            // the reparameterized counterpart of {computePotentialAndGradient}: the data
            // gradient term is scaled elementwise by {jacobian} before being combined,
            // grad_potential[s, :] = -(grad_prior[s, :] + beta * jacobian[s, :] * grad_data[s, :])
            template <typename realtype_t>
            void computePotentialAndGradientReparam(
                vector_view_t<realtype_t, true> prior,
                vector_view_t<realtype_t, true> data,
                matrix_view_t<realtype_t, true> grad_prior,
                matrix_view_t<realtype_t, true> grad_data,
                matrix_view_t<realtype_t, true> jacobian,
                vector_view_t<realtype_t, false> potential,
                matrix_view_t<realtype_t, false> grad_potential,
                const realtype_t beta,
                cudaStream_t stream=0);

            // kinetic[s] = 0.5 * ||momentum[s, :]||^2
            template <typename realtype_t>
            void kineticEnergy(
                matrix_view_t<realtype_t, true> momentum,
                vector_view_t<realtype_t, false> kinetic,
                cudaStream_t stream=0);

            // theta += step * momentum, elementwise, in place
            template <typename realtype_t>
            void updatePosition(
                matrix_view_t<realtype_t, false> theta,
                matrix_view_t<realtype_t, true> momentum,
                const realtype_t step, cudaStream_t stream=0);

            // momentum += scale * gradU, elementwise, in place
            template <typename realtype_t>
            void updateMomentum(
                matrix_view_t<realtype_t, false> momentum,
                matrix_view_t<realtype_t, true> gradU,
                const realtype_t scale, cudaStream_t stream=0);

            // one Metropolis-Hastings accept/reject test per sample: mask[s] = 1 if
            // log(u) < -deltaH[s] for a fresh uniform draw u, else 0
            template <typename realtype_t>
            void metropolis(
                vector_view_t<realtype_t, true> deltaH,
                vector_view_t<int> mask,
                cudaStream_t stream=0);

            // restore both {theta}/{momentum} to their {*_old} values, row by row, for every
            // sample where {mask[sample] == 0} (rejected)
            template <typename realtype_t>
            void restoreRejected(
                matrix_view_t<realtype_t, false> theta,
                matrix_view_t<realtype_t, true> theta_old,
                matrix_view_t<realtype_t, false> momentum,
                matrix_view_t<realtype_t, true> momentum_old,
                vector_view_t<int, true> mask,
                cudaStream_t stream=0);

            // the single-matrix counterpart of {restoreRejected}
            template <typename realtype_t>
            void restoreMatrix(
                matrix_view_t<realtype_t, false> current,
                matrix_view_t<realtype_t, true> backup,
                vector_view_t<int, true> mask,
                cudaStream_t stream=0);

        } // namespace cudaLeapfrog
    } // namespace bayesian
}} // namespace altar::cuda


#endif //altar_cuda_bayesian_cudaLeapfrog_h
// end of file
