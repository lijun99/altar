// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//

// declarations
#include "cudaLangevin.h"
// cuda utitlities
// shared NTHREADS/IDIVUP/cudaCheckError/matrix_view_t/vector_view_t
#include "../support.h"

#include <pyre/cuda.h>
#include <iostream>

// cuda kernels; defined here, ahead of the launcher functions below that instantiate and
// launch them (see cudaL2.cu/cudaGaussian.cu for why the ordering matters)
namespace cudaLangevin_kernels {

    // the SGLD update, one parameter at a time, one thread per sample
    template <typename realtype_t>
    __global__ void
    updateTheta_kernel(matrix_view_t<realtype_t, false> theta,
        vector_view_t<realtype_t, true> prior_gradient,
        vector_view_t<realtype_t, true> datalikelihood_gradient,
        const realtype_t half_epsilon_t, vector_view_t<realtype_t, true> eta_t,
        const int index)
    {
        int sample = blockIdx.x*blockDim.x + threadIdx.x;
        auto samples = theta.packing().shape()[0];
        if (sample >= samples) return;

        theta[{ sample, index }] +=
            half_epsilon_t * (prior_gradient[{ sample }] + datalikelihood_gradient[{ sample }])
            + eta_t[{ sample }];
    }

    // the batched SGLD update, every parameter at once, one thread per sample
    template <typename realtype_t>
    __global__ void
    updateThetaBatched_kernel(matrix_view_t<realtype_t, false> theta,
        const realtype_t alpha1, matrix_view_t<realtype_t, true> prior_gradient,
        const realtype_t alpha2, matrix_view_t<realtype_t, true> datalikelihood_gradient,
        const realtype_t half_epsilon_t, matrix_view_t<realtype_t, true> eta_t)
    {
        int sample = blockIdx.x*blockDim.x + threadIdx.x;
        auto samples = theta.packing().shape()[0];
        auto parameters = theta.packing().shape()[1];
        if (sample >= samples) return;

        for (int parameter = 0; parameter < parameters; ++parameter) {
            theta[{ sample, parameter }] +=
                half_epsilon_t * (alpha1*prior_gradient[{ sample, parameter }]
                    + alpha2*datalikelihood_gradient[{ sample, parameter }])
                + eta_t[{ sample, parameter }];
        }
    }

} // of namespace cudaLangevin_kernels


// launch {cudaLangevin_kernels::updateTheta_kernel}
template <typename realtype_t>
void altar::cuda::bayesian::cudaLangevin::
updateTheta(matrix_view_t<realtype_t, false> theta,
    vector_view_t<realtype_t, true> prior_gradient,
    vector_view_t<realtype_t, true> datalikelihood_gradient,
    const realtype_t half_epsilon_t, vector_view_t<realtype_t, true> eta_t,
    const size_t index,
    cudaStream_t stream)
{
    auto samples = theta.packing().shape()[0];
    auto blockSize = NTHREADS;
    auto gridSize = IDIVUP(samples, blockSize);
    cudaLangevin_kernels::updateTheta_kernel<realtype_t><<<gridSize, blockSize, 0, stream>>>(
        theta, prior_gradient, datalikelihood_gradient, half_epsilon_t, eta_t, static_cast<int>(index));
    cudaCheckError("cudaLangevin::updateTheta error");
}

template void altar::cuda::bayesian::cudaLangevin::updateTheta<float>(
    matrix_view_t<float, false>, vector_view_t<float, true>, vector_view_t<float, true>,
    const float, vector_view_t<float, true>, const size_t, cudaStream_t);
template void altar::cuda::bayesian::cudaLangevin::updateTheta<double>(
    matrix_view_t<double, false>, vector_view_t<double, true>, vector_view_t<double, true>,
    const double, vector_view_t<double, true>, const size_t, cudaStream_t);


// launch {cudaLangevin_kernels::updateThetaBatched_kernel}
template <typename realtype_t>
void altar::cuda::bayesian::cudaLangevin::
updateThetaBatched(matrix_view_t<realtype_t, false> theta,
    const realtype_t alpha1, matrix_view_t<realtype_t, true> prior_gradient,
    const realtype_t alpha2, matrix_view_t<realtype_t, true> datalikelihood_gradient,
    const realtype_t half_epsilon_t, matrix_view_t<realtype_t, true> eta_t,
    cudaStream_t stream)
{
    auto samples = theta.packing().shape()[0];
    auto blockSize = NTHREADS;
    auto gridSize = IDIVUP(samples, blockSize);
    cudaLangevin_kernels::updateThetaBatched_kernel<realtype_t><<<gridSize, blockSize, 0, stream>>>(
        theta, alpha1, prior_gradient, alpha2, datalikelihood_gradient, half_epsilon_t, eta_t);
    cudaCheckError("cudaLangevin::updateThetaBatched error");
}

template void altar::cuda::bayesian::cudaLangevin::updateThetaBatched<float>(
    matrix_view_t<float, false>, const float, matrix_view_t<float, true>,
    const float, matrix_view_t<float, true>, const float, matrix_view_t<float, true>, cudaStream_t);
template void altar::cuda::bayesian::cudaLangevin::updateThetaBatched<double>(
    matrix_view_t<double, false>, const double, matrix_view_t<double, true>,
    const double, matrix_view_t<double, true>, const double, matrix_view_t<double, true>, cudaStream_t);

// end of file
