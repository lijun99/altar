// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//
// Author(s): Lijun Zhu, Codex

//! file cudaLeapfrog.cu
//! Leapfrog integrator support for Hamiltonian Monte Carlo

#include "cudaLeapfrog.h"

#include <pyre/cuda.h>
#include <curand_kernel.h>
#include <altar/cuda/distributions/cudaRandom.h>

#include <chrono>
#include <math.h>

namespace {
    inline unsigned long long current_seed() {
        return static_cast<unsigned long long>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count());
    }
}

namespace cudaLeapfrog_kernels {

    template <typename realtype_t>
    __global__ void sampleMomentumKernel(realtype_t * const momentum,
                                         const size_t total, const unsigned long long seed)
    {
        const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= total) {
            return;
        }

        curandState state;
        curand_init(seed, static_cast<unsigned long long>(tid), 0, &state);
        momentum[tid] = altar::cuda::distributions::curandNormal<realtype_t>(&state);
    }

    template <typename realtype_t>
    __global__ void potentialKernel(const realtype_t * const prior,
                                    const realtype_t * const data,
                                    realtype_t * const potential,
                                    const size_t samples, const realtype_t beta)
    {
        const size_t sid = blockIdx.x * blockDim.x + threadIdx.x;
        if (sid >= samples) {
            return;
        }
        potential[sid] = - (prior[sid] + beta * data[sid]);
    }

    template <typename realtype_t, bool hasJacobian>
    __global__ void gradientKernel(const realtype_t * const grad_prior,
                                   const realtype_t * const grad_data,
                                   const realtype_t * const jacobian,
                                   realtype_t * const grad_potential,
                                   const size_t total, const size_t parameters,
                                   const realtype_t beta)
    {
        const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= total) {
            return;
        }
        if constexpr (hasJacobian) {
            const realtype_t factor = (jacobian ? jacobian[tid] : static_cast<realtype_t>(1));
            grad_potential[tid] = - (grad_prior[tid] + beta * factor * grad_data[tid]);
        } else {
            (void) parameters;
            grad_potential[tid] = - (grad_prior[tid] + beta * grad_data[tid]);
        }
    }

    template <typename realtype_t>
    __global__ void kineticKernel(const realtype_t * const momentum,
                                  realtype_t * const kinetic,
                                  const size_t samples, const size_t parameters)
    {
        const size_t sid = blockIdx.x * blockDim.x + threadIdx.x;
        if (sid >= samples) {
            return;
        }
        const realtype_t * const sample_momentum = momentum + sid * parameters;
        realtype_t sum = 0;
        for (size_t p = 0; p < parameters; ++p) {
            const realtype_t value = sample_momentum[p];
            sum += value * value;
        }
        kinetic[sid] = static_cast<realtype_t>(0.5) * sum;
    }

    template <typename realtype_t>
    __global__ void updatePositionKernel(realtype_t * const theta,
                                         const realtype_t * const momentum,
                                         const realtype_t step,
                                         const size_t total)
    {
        const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= total) {
            return;
        }
        theta[tid] += step * momentum[tid];
    }

    template <typename realtype_t>
    __global__ void updateMomentumKernel(realtype_t * const momentum,
                                         const realtype_t * const grad,
                                         const realtype_t scale,
                                         const size_t total)
    {
        const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= total) {
            return;
        }
        momentum[tid] += scale * grad[tid];
    }

    template <typename realtype_t>
    __global__ void metropolisKernel(const realtype_t * const deltaH,
                                     int * const mask,
                                     const size_t samples,
                                     const unsigned long long seed)
    {
        const size_t sid = blockIdx.x * blockDim.x + threadIdx.x;
        if (sid >= samples) {
            return;
        }
        curandState state;
        curand_init(seed, static_cast<unsigned long long>(sid), 0, &state);
        const double u = altar::cuda::distributions::curandUniform<double>(&state);
        const double logu = log(u);
        mask[sid] = (logu < -static_cast<double>(deltaH[sid])) ? 1 : 0;
    }

    template <typename realtype_t>
    __global__ void restoreKernel(realtype_t * const current,
                                  const realtype_t * const backup,
                                  const int * const mask,
                                  const size_t samples, const size_t parameters)
    {
        const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        const size_t total = samples * parameters;
        if (tid >= total) {
            return;
        }
        const size_t sid = tid / parameters;
        if (!mask[sid]) {
            current[tid] = backup[tid];
        }
    }
}


namespace altar { namespace cuda { namespace bayesian { namespace cudaLeapfrog {

    template <typename realtype_t>
    void sampleMomentum(realtype_t * const momentum,
                        const size_t samples, const size_t parameters,
                        cudaStream_t stream)
    {
        if (samples == 0 || parameters == 0) {
            return;
        }
        const size_t total = samples * parameters;
        const int blockSize = NTHREADS;
        const int gridSize = IDIVUP(static_cast<int>(total), blockSize);
        const unsigned long long seed = current_seed();
        cudaLeapfrog_kernels::sampleMomentumKernel<realtype_t>
            <<<gridSize, blockSize, 0, stream>>>(momentum, total, seed);
        cudaCheckError("cudaLeapfrog:sampleMomentum Error");
    }

    template void sampleMomentum<float>(float * const, const size_t, const size_t, cudaStream_t);
    template void sampleMomentum<double>(double * const, const size_t, const size_t, cudaStream_t);


    template <typename realtype_t>
    void computePotentialAndGradient(const realtype_t * const prior,
                                     const realtype_t * const data,
                                     const realtype_t * const grad_prior,
                                     const realtype_t * const grad_data,
                                     realtype_t * const potential,
                                     realtype_t * const grad_potential,
                                     const realtype_t * const jacobian,
                                     const size_t samples, const size_t parameters,
                                     const realtype_t beta, const bool reparameterization,
                                     cudaStream_t stream)
    {
        if (samples == 0 || parameters == 0) {
            return;
        }
        const int blockSize = NTHREADS;
        const int gridPot = IDIVUP(static_cast<int>(samples), blockSize);
        cudaLeapfrog_kernels::potentialKernel<realtype_t>
            <<<gridPot, blockSize, 0, stream>>>(prior, data, potential, samples, beta);
        cudaCheckError("cudaLeapfrog:potentialKernel Error");

        const size_t total = samples * parameters;
        const int gridGrad = IDIVUP(static_cast<int>(total), blockSize);
        if (reparameterization && jacobian) {
            cudaLeapfrog_kernels::gradientKernel<realtype_t, true>
                <<<gridGrad, blockSize, 0, stream>>>(grad_prior, grad_data, jacobian,
                                                     grad_potential, total, parameters, beta);
        } else {
            cudaLeapfrog_kernels::gradientKernel<realtype_t, false>
                <<<gridGrad, blockSize, 0, stream>>>(grad_prior, grad_data, jacobian,
                                                     grad_potential, total, parameters, beta);
        }
        cudaCheckError("cudaLeapfrog:gradientKernel Error");
    }

    template void computePotentialAndGradient<float>(const float *, const float *, const float *, const float *, float *, float *, const float *, const size_t, const size_t, const float, const bool, cudaStream_t);
    template void computePotentialAndGradient<double>(const double *, const double *, const double *, const double *, double *, double *, const double *, const size_t, const size_t, const double, const bool, cudaStream_t);


    template <typename realtype_t>
    void kineticEnergy(const realtype_t * const momentum,
                       realtype_t * const kinetic,
                       const size_t samples, const size_t parameters,
                       cudaStream_t stream)
    {
        if (samples == 0) {
            return;
        }
        const int blockSize = NTHREADS;
        const int gridSize = IDIVUP(static_cast<int>(samples), blockSize);
        cudaLeapfrog_kernels::kineticKernel<realtype_t>
            <<<gridSize, blockSize, 0, stream>>>(momentum, kinetic, samples, parameters);
        cudaCheckError("cudaLeapfrog:kineticKernel Error");
    }

    template void kineticEnergy<float>(const float *, float *, const size_t, const size_t, cudaStream_t);
    template void kineticEnergy<double>(const double *, double *, const size_t, const size_t, cudaStream_t);


    template <typename realtype_t>
    void updatePosition(realtype_t * const theta,
                        const realtype_t * const momentum,
                        const size_t samples, const size_t parameters,
                        const realtype_t step, cudaStream_t stream)
    {
        const size_t total = samples * parameters;
        if (total == 0) {
            return;
        }
        const int blockSize = NTHREADS;
        const int gridSize = IDIVUP(static_cast<int>(total), blockSize);
        cudaLeapfrog_kernels::updatePositionKernel<realtype_t>
            <<<gridSize, blockSize, 0, stream>>>(theta, momentum, step, total);
        cudaCheckError("cudaLeapfrog:updatePosition Error");
    }

    template void updatePosition<float>(float * const, const float * const, const size_t, const size_t, const float, cudaStream_t);
    template void updatePosition<double>(double * const, const double * const, const size_t, const size_t, const double, cudaStream_t);


    template <typename realtype_t>
    void updateMomentum(realtype_t * const momentum,
                        const realtype_t * const gradU,
                        const size_t samples, const size_t parameters,
                        const realtype_t scale,
                        cudaStream_t stream)
    {
        const size_t total = samples * parameters;
        if (total == 0) {
            return;
        }
        const int blockSize = NTHREADS;
        const int gridSize = IDIVUP(static_cast<int>(total), blockSize);
        cudaLeapfrog_kernels::updateMomentumKernel<realtype_t>
            <<<gridSize, blockSize, 0, stream>>>(momentum, gradU, scale, total);
        cudaCheckError("cudaLeapfrog:updateMomentum Error");
    }

    template void updateMomentum<float>(float * const, const float * const, const size_t, const size_t, const float, cudaStream_t);
    template void updateMomentum<double>(double * const, const double * const, const size_t, const size_t, const double, cudaStream_t);


    template <typename realtype_t>
    void metropolis(const realtype_t * const deltaH,
                    int * const mask,
                    const size_t samples,
                    cudaStream_t stream)
    {
        if (samples == 0) {
            return;
        }
        const int blockSize = NTHREADS;
        const int gridSize = IDIVUP(static_cast<int>(samples), blockSize);
        const unsigned long long seed = current_seed();
        cudaLeapfrog_kernels::metropolisKernel<realtype_t>
            <<<gridSize, blockSize, 0, stream>>>(deltaH, mask, samples, seed);
        cudaCheckError("cudaLeapfrog:metropolisKernel Error");
    }

    template void metropolis<float>(const float * const, int * const, const size_t, cudaStream_t);
    template void metropolis<double>(const double * const, int * const, const size_t, cudaStream_t);


    template <typename realtype_t>
    void restoreRejected(realtype_t * const theta,
                         const realtype_t * const theta_old,
                         realtype_t * const momentum,
                         const realtype_t * const momentum_old,
                         const int * const mask,
                         const size_t samples, const size_t parameters,
                         cudaStream_t stream)
    {
        if (samples == 0 || parameters == 0) {
            return;
        }
        const size_t total = samples * parameters;
        const int blockSize = NTHREADS;
        const int gridSize = IDIVUP(static_cast<int>(total), blockSize);
        cudaLeapfrog_kernels::restoreKernel<realtype_t>
            <<<gridSize, blockSize, 0, stream>>>(theta, theta_old, mask, samples, parameters);
        cudaLeapfrog_kernels::restoreKernel<realtype_t>
            <<<gridSize, blockSize, 0, stream>>>(momentum, momentum_old, mask, samples, parameters);
        cudaCheckError("cudaLeapfrog:restoreKernel Error");
    }

    template void restoreRejected<float>(float * const, const float * const, float * const, const float * const, const int * const, const size_t, const size_t, cudaStream_t);
    template void restoreRejected<double>(double * const, const double * const, double * const, const double * const, const int * const, const size_t, const size_t, cudaStream_t);


    template <typename realtype_t>
    void restoreMatrix(realtype_t * const current,
                       const realtype_t * const backup,
                       const int * const mask,
                       const size_t samples, const size_t parameters,
                       cudaStream_t stream)
    {
        if (samples == 0 || parameters == 0) {
            return;
        }
        const size_t total = samples * parameters;
        const int blockSize = NTHREADS;
        const int gridSize = IDIVUP(static_cast<int>(total), blockSize);
        cudaLeapfrog_kernels::restoreKernel<realtype_t>
            <<<gridSize, blockSize, 0, stream>>>(current, backup, mask, samples, parameters);
        cudaCheckError("cudaLeapfrog:restoreMatrix Error");
    }

    template void restoreMatrix<float>(float * const, const float * const, const int * const, const size_t, const size_t, cudaStream_t);
    template void restoreMatrix<double>(double * const, const double * const, const int * const, const size_t, const size_t, cudaStream_t);

}}}} // namespace altar::cuda::bayesian::cudaLeapfrog

// end of file
