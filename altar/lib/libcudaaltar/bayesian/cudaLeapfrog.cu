// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//

//! file cudaLeapfrog.cu
//! Leapfrog integrator support for Hamiltonian Monte Carlo

#include "cudaLeapfrog.h"

// shared NTHREADS/IDIVUP/cudaCheckError/matrix_view_t/vector_view_t
#include "../support.h"

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

    // fill {momentum} with iid draws from N(0, 1), one thread per (sample, parameter) cell
    template <typename realtype_t>
    __global__ void sampleMomentumKernel(matrix_view_t<realtype_t, false> momentum,
                                         const unsigned long long seed)
    {
        auto samples = momentum.packing().shape()[0];
        auto parameters = momentum.packing().shape()[1];
        auto total = samples * parameters;
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= total) return;

        curandState state;
        curand_init(seed, static_cast<unsigned long long>(tid), 0, &state);
        int sid = tid / parameters;
        int pid = tid % parameters;
        momentum[{ sid, pid }] = altar::cuda::distributions::curandNormal<realtype_t>(&state);
    }

    // potential[s] = -(prior[s] + beta * data[s]), one thread per sample
    template <typename realtype_t>
    __global__ void potentialKernel(vector_view_t<realtype_t, true> prior,
                                    vector_view_t<realtype_t, true> data,
                                    vector_view_t<realtype_t, false> potential,
                                    const realtype_t beta)
    {
        auto samples = potential.packing().shape()[0];
        int sid = blockIdx.x * blockDim.x + threadIdx.x;
        if (sid >= samples) return;
        potential[{ sid }] = -(prior[{ sid }] + beta * data[{ sid }]);
    }

    // grad_potential[s, :] = -(grad_prior[s, :] + beta * grad_data[s, :]) when {hasJacobian}
    // is false, or -(grad_prior[s, :] + beta * jacobian[s, :] * grad_data[s, :]) when true;
    // one thread per (sample, parameter) cell
    template <typename realtype_t, bool hasJacobian>
    __global__ void gradientKernel(matrix_view_t<realtype_t, true> grad_prior,
                                   matrix_view_t<realtype_t, true> grad_data,
                                   matrix_view_t<realtype_t, true> jacobian,
                                   matrix_view_t<realtype_t, false> grad_potential,
                                   const realtype_t beta)
    {
        auto samples = grad_potential.packing().shape()[0];
        auto parameters = grad_potential.packing().shape()[1];
        auto total = samples * parameters;
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= total) return;
        int sid = tid / parameters;
        int pid = tid % parameters;

        if constexpr (hasJacobian) {
            grad_potential[{ sid, pid }] =
                -(grad_prior[{ sid, pid }] + beta * jacobian[{ sid, pid }] * grad_data[{ sid, pid }]);
        } else {
            grad_potential[{ sid, pid }] = -(grad_prior[{ sid, pid }] + beta * grad_data[{ sid, pid }]);
        }
    }

    // kinetic[s] = 0.5 * ||momentum[s, :]||^2, one thread per sample
    template <typename realtype_t>
    __global__ void kineticKernel(matrix_view_t<realtype_t, true> momentum,
                                  vector_view_t<realtype_t, false> kinetic)
    {
        auto samples = momentum.packing().shape()[0];
        auto parameters = momentum.packing().shape()[1];
        int sid = blockIdx.x * blockDim.x + threadIdx.x;
        if (sid >= samples) return;

        auto sum = realtype_t{ 0 };
        for (int p = 0; p < parameters; ++p) {
            auto value = momentum[{ sid, p }];
            sum += value * value;
        }
        kinetic[{ sid }] = realtype_t{ 0.5 } * sum;
    }

    // theta += step * momentum, elementwise, one thread per (sample, parameter) cell
    template <typename realtype_t>
    __global__ void updatePositionKernel(matrix_view_t<realtype_t, false> theta,
                                         matrix_view_t<realtype_t, true> momentum,
                                         const realtype_t step)
    {
        auto samples = theta.packing().shape()[0];
        auto parameters = theta.packing().shape()[1];
        auto total = samples * parameters;
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= total) return;
        int sid = tid / parameters;
        int pid = tid % parameters;
        theta[{ sid, pid }] += step * momentum[{ sid, pid }];
    }

    // momentum += scale * grad, elementwise, one thread per (sample, parameter) cell
    template <typename realtype_t>
    __global__ void updateMomentumKernel(matrix_view_t<realtype_t, false> momentum,
                                         matrix_view_t<realtype_t, true> grad,
                                         const realtype_t scale)
    {
        auto samples = momentum.packing().shape()[0];
        auto parameters = momentum.packing().shape()[1];
        auto total = samples * parameters;
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= total) return;
        int sid = tid / parameters;
        int pid = tid % parameters;
        momentum[{ sid, pid }] += scale * grad[{ sid, pid }];
    }

    // one Metropolis-Hastings accept/reject test per sample
    template <typename realtype_t>
    __global__ void metropolisKernel(vector_view_t<realtype_t, true> deltaH,
                                     vector_view_t<int> mask,
                                     const unsigned long long seed)
    {
        auto samples = mask.packing().shape()[0];
        int sid = blockIdx.x * blockDim.x + threadIdx.x;
        if (sid >= samples) return;

        curandState state;
        curand_init(seed, static_cast<unsigned long long>(sid), 0, &state);
        auto u = altar::cuda::distributions::curandUniform<double>(&state);
        auto logu = log(u);
        mask[{ sid }] = (logu < -static_cast<double>(deltaH[{ sid }])) ? 1 : 0;
    }

    // restore every row of {current} where {mask[sample] == 0} (rejected) to {backup}'s;
    // one thread per (sample, parameter) cell
    template <typename realtype_t>
    __global__ void restoreKernel(matrix_view_t<realtype_t, false> current,
                                  matrix_view_t<realtype_t, true> backup,
                                  vector_view_t<int, true> mask)
    {
        auto samples = current.packing().shape()[0];
        auto parameters = current.packing().shape()[1];
        auto total = samples * parameters;
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= total) return;
        int sid = tid / parameters;
        int pid = tid % parameters;
        if (!mask[{ sid }]) {
            current[{ sid, pid }] = backup[{ sid, pid }];
        }
    }

    // restore {current[sample]} to {backup[sample]} wherever {mask[sample] == 0}; one thread per sample
    template <typename realtype_t>
    __global__ void restoreVectorKernel(vector_view_t<realtype_t, false> current,
                                        vector_view_t<realtype_t, true> backup,
                                        vector_view_t<int, true> mask)
    {
        auto samples = current.packing().shape()[0];
        int sid = blockIdx.x * blockDim.x + threadIdx.x;
        if (sid >= samples) return;
        if (!mask[{ sid }]) {
            current[{ sid }] = backup[{ sid }];
        }
    }

} // of namespace cudaLeapfrog_kernels


namespace altar { namespace cuda { namespace bayesian { namespace cudaLeapfrog {

    template <typename realtype_t>
    void sampleMomentum(matrix_view_t<realtype_t, false> momentum, cudaStream_t stream)
    {
        auto samples = momentum.packing().shape()[0];
        auto parameters = momentum.packing().shape()[1];
        auto total = samples * parameters;
        if (total == 0) return;

        auto blockSize = NTHREADS;
        auto gridSize = IDIVUP(total, blockSize);
        auto seed = current_seed();
        cudaLeapfrog_kernels::sampleMomentumKernel<realtype_t>
            <<<gridSize, blockSize, 0, stream>>>(momentum, seed);
        cudaCheckError("cudaLeapfrog::sampleMomentum error");
    }

    template void sampleMomentum<float>(matrix_view_t<float, false>, cudaStream_t);
    template void sampleMomentum<double>(matrix_view_t<double, false>, cudaStream_t);


    template <typename realtype_t>
    void computePotentialAndGradient(
        vector_view_t<realtype_t, true> prior,
        vector_view_t<realtype_t, true> data,
        matrix_view_t<realtype_t, true> grad_prior,
        matrix_view_t<realtype_t, true> grad_data,
        vector_view_t<realtype_t, false> potential,
        matrix_view_t<realtype_t, false> grad_potential,
        const realtype_t beta,
        cudaStream_t stream)
    {
        auto samples = potential.packing().shape()[0];
        auto parameters = grad_potential.packing().shape()[1];
        if (samples == 0 || parameters == 0) return;

        auto blockSize = NTHREADS;
        auto gridPot = IDIVUP(samples, blockSize);
        cudaLeapfrog_kernels::potentialKernel<realtype_t>
            <<<gridPot, blockSize, 0, stream>>>(prior, data, potential, beta);
        cudaCheckError("cudaLeapfrog::potentialKernel error");

        auto total = samples * parameters;
        auto gridGrad = IDIVUP(total, blockSize);
        // {jacobian} is unused on this path; {grad_data} doubles as a placeholder argument
        // (the {hasJacobian=false} kernel instantiation never reads it)
        cudaLeapfrog_kernels::gradientKernel<realtype_t, false>
            <<<gridGrad, blockSize, 0, stream>>>(grad_prior, grad_data, grad_data, grad_potential, beta);
        cudaCheckError("cudaLeapfrog::gradientKernel error");
    }

    template void computePotentialAndGradient<float>(
        vector_view_t<float, true>, vector_view_t<float, true>, matrix_view_t<float, true>, matrix_view_t<float, true>,
        vector_view_t<float, false>, matrix_view_t<float, false>, const float, cudaStream_t);
    template void computePotentialAndGradient<double>(
        vector_view_t<double, true>, vector_view_t<double, true>, matrix_view_t<double, true>, matrix_view_t<double, true>,
        vector_view_t<double, false>, matrix_view_t<double, false>, const double, cudaStream_t);


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
        cudaStream_t stream)
    {
        auto samples = potential.packing().shape()[0];
        auto parameters = grad_potential.packing().shape()[1];
        if (samples == 0 || parameters == 0) return;

        auto blockSize = NTHREADS;
        auto gridPot = IDIVUP(samples, blockSize);
        cudaLeapfrog_kernels::potentialKernel<realtype_t>
            <<<gridPot, blockSize, 0, stream>>>(prior, data, potential, beta);
        cudaCheckError("cudaLeapfrog::potentialKernel error");

        auto total = samples * parameters;
        auto gridGrad = IDIVUP(total, blockSize);
        cudaLeapfrog_kernels::gradientKernel<realtype_t, true>
            <<<gridGrad, blockSize, 0, stream>>>(grad_prior, grad_data, jacobian, grad_potential, beta);
        cudaCheckError("cudaLeapfrog::gradientKernel error");
    }

    template void computePotentialAndGradientReparam<float>(
        vector_view_t<float, true>, vector_view_t<float, true>, matrix_view_t<float, true>, matrix_view_t<float, true>,
        matrix_view_t<float, true>, vector_view_t<float, false>, matrix_view_t<float, false>, const float, cudaStream_t);
    template void computePotentialAndGradientReparam<double>(
        vector_view_t<double, true>, vector_view_t<double, true>, matrix_view_t<double, true>, matrix_view_t<double, true>,
        matrix_view_t<double, true>, vector_view_t<double, false>, matrix_view_t<double, false>, const double, cudaStream_t);


    template <typename realtype_t>
    void kineticEnergy(matrix_view_t<realtype_t, true> momentum, vector_view_t<realtype_t, false> kinetic,
                       cudaStream_t stream)
    {
        auto samples = kinetic.packing().shape()[0];
        if (samples == 0) return;

        auto blockSize = NTHREADS;
        auto gridSize = IDIVUP(samples, blockSize);
        cudaLeapfrog_kernels::kineticKernel<realtype_t>
            <<<gridSize, blockSize, 0, stream>>>(momentum, kinetic);
        cudaCheckError("cudaLeapfrog::kineticEnergy error");
    }

    template void kineticEnergy<float>(matrix_view_t<float, true>, vector_view_t<float, false>, cudaStream_t);
    template void kineticEnergy<double>(matrix_view_t<double, true>, vector_view_t<double, false>, cudaStream_t);


    template <typename realtype_t>
    void updatePosition(matrix_view_t<realtype_t, false> theta, matrix_view_t<realtype_t, true> momentum,
                        const realtype_t step, cudaStream_t stream)
    {
        auto samples = theta.packing().shape()[0];
        auto parameters = theta.packing().shape()[1];
        auto total = samples * parameters;
        if (total == 0) return;

        auto blockSize = NTHREADS;
        auto gridSize = IDIVUP(total, blockSize);
        cudaLeapfrog_kernels::updatePositionKernel<realtype_t>
            <<<gridSize, blockSize, 0, stream>>>(theta, momentum, step);
        cudaCheckError("cudaLeapfrog::updatePosition error");
    }

    template void updatePosition<float>(matrix_view_t<float, false>, matrix_view_t<float, true>, const float, cudaStream_t);
    template void updatePosition<double>(matrix_view_t<double, false>, matrix_view_t<double, true>, const double, cudaStream_t);


    template <typename realtype_t>
    void updateMomentum(matrix_view_t<realtype_t, false> momentum, matrix_view_t<realtype_t, true> gradU,
                        const realtype_t scale, cudaStream_t stream)
    {
        auto samples = momentum.packing().shape()[0];
        auto parameters = momentum.packing().shape()[1];
        auto total = samples * parameters;
        if (total == 0) return;

        auto blockSize = NTHREADS;
        auto gridSize = IDIVUP(total, blockSize);
        cudaLeapfrog_kernels::updateMomentumKernel<realtype_t>
            <<<gridSize, blockSize, 0, stream>>>(momentum, gradU, scale);
        cudaCheckError("cudaLeapfrog::updateMomentum error");
    }

    template void updateMomentum<float>(matrix_view_t<float, false>, matrix_view_t<float, true>, const float, cudaStream_t);
    template void updateMomentum<double>(matrix_view_t<double, false>, matrix_view_t<double, true>, const double, cudaStream_t);


    template <typename realtype_t>
    void metropolis(vector_view_t<realtype_t, true> deltaH, vector_view_t<int> mask, cudaStream_t stream)
    {
        auto samples = mask.packing().shape()[0];
        if (samples == 0) return;

        auto blockSize = NTHREADS;
        auto gridSize = IDIVUP(samples, blockSize);
        auto seed = current_seed();
        cudaLeapfrog_kernels::metropolisKernel<realtype_t>
            <<<gridSize, blockSize, 0, stream>>>(deltaH, mask, seed);
        cudaCheckError("cudaLeapfrog::metropolis error");
    }

    template void metropolis<float>(vector_view_t<float, true>, vector_view_t<int>, cudaStream_t);
    template void metropolis<double>(vector_view_t<double, true>, vector_view_t<int>, cudaStream_t);


    template <typename realtype_t>
    void restoreRejected(
        matrix_view_t<realtype_t, false> theta, matrix_view_t<realtype_t, true> theta_old,
        matrix_view_t<realtype_t, false> momentum, matrix_view_t<realtype_t, true> momentum_old,
        vector_view_t<int, true> mask, cudaStream_t stream)
    {
        auto samples = theta.packing().shape()[0];
        auto parameters = theta.packing().shape()[1];
        auto total = samples * parameters;
        if (total == 0) return;

        auto blockSize = NTHREADS;
        auto gridSize = IDIVUP(total, blockSize);
        cudaLeapfrog_kernels::restoreKernel<realtype_t>
            <<<gridSize, blockSize, 0, stream>>>(theta, theta_old, mask);
        cudaLeapfrog_kernels::restoreKernel<realtype_t>
            <<<gridSize, blockSize, 0, stream>>>(momentum, momentum_old, mask);
        cudaCheckError("cudaLeapfrog::restoreRejected error");
    }

    template void restoreRejected<float>(
        matrix_view_t<float, false>, matrix_view_t<float, true>, matrix_view_t<float, false>, matrix_view_t<float, true>,
        vector_view_t<int, true>, cudaStream_t);
    template void restoreRejected<double>(
        matrix_view_t<double, false>, matrix_view_t<double, true>, matrix_view_t<double, false>, matrix_view_t<double, true>,
        vector_view_t<int, true>, cudaStream_t);


    template <typename realtype_t>
    void restoreMatrix(matrix_view_t<realtype_t, false> current, matrix_view_t<realtype_t, true> backup,
                       vector_view_t<int, true> mask, cudaStream_t stream)
    {
        auto samples = current.packing().shape()[0];
        auto parameters = current.packing().shape()[1];
        auto total = samples * parameters;
        if (total == 0) return;

        auto blockSize = NTHREADS;
        auto gridSize = IDIVUP(total, blockSize);
        cudaLeapfrog_kernels::restoreKernel<realtype_t>
            <<<gridSize, blockSize, 0, stream>>>(current, backup, mask);
        cudaCheckError("cudaLeapfrog::restoreMatrix error");
    }

    template void restoreMatrix<float>(matrix_view_t<float, false>, matrix_view_t<float, true>, vector_view_t<int, true>, cudaStream_t);
    template void restoreMatrix<double>(matrix_view_t<double, false>, matrix_view_t<double, true>, vector_view_t<int, true>, cudaStream_t);


    template <typename realtype_t>
    void restoreVector(vector_view_t<realtype_t, false> current, vector_view_t<realtype_t, true> backup,
                       vector_view_t<int, true> mask, cudaStream_t stream)
    {
        auto samples = current.packing().shape()[0];
        if (samples == 0) return;

        auto blockSize = NTHREADS;
        auto gridSize = IDIVUP(samples, blockSize);
        cudaLeapfrog_kernels::restoreVectorKernel<realtype_t>
            <<<gridSize, blockSize, 0, stream>>>(current, backup, mask);
        cudaCheckError("cudaLeapfrog::restoreVector error");
    }

    template void restoreVector<float>(vector_view_t<float, false>, vector_view_t<float, true>, vector_view_t<int, true>, cudaStream_t);
    template void restoreVector<double>(vector_view_t<double, false>, vector_view_t<double, true>, vector_view_t<int, true>, cudaStream_t);

}}}} // namespace altar::cuda::bayesian::cudaLeapfrog

// end of file
