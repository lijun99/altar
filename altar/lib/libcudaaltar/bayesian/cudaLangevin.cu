// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-2021 parasim inc
// (c) 2010-2021 california institute of technology
// all rights reserved
//
// Author(s): Lijun Zhu

// declarations
#include "cudaLangevin.h"
// cuda utitlities
#include <pyre/cuda.h>
#include <iostream>

// cuda kernel declarations
namespace cudaLangevin_kernels {

    template <typename realtype_t>
    __global__ void  updateTheta_kernel(realtype_t * const theta,
                const realtype_t * const prior_gradient,
                const realtype_t * const datalikelihood_gradient,
                const realtype_t half_epsilon_t, const realtype_t * const eta_t,
                const size_t samples, const size_t parameters, const size_t index);

    template <typename realtype_t>
    __global__ void  updateThetaBatched_kernel(realtype_t * const theta,
                const realtype_t alpha1, const realtype_t * const prior_gradient,
                const realtype_t alpha2, const realtype_t * const datalikelihood_gradient,
                const realtype_t half_epsilon_t, const realtype_t * const eta_t,
                const size_t samples, const size_t parameters);

}


template <typename realtype_t>
void altar::cuda::bayesian::cudaLangevin::
updateTheta(realtype_t * const theta,
                const realtype_t * const prior_gradient,
                const realtype_t * const datalikelihood_gradient,
                const realtype_t half_epsilon_t, const realtype_t * const eta_t,
                const size_t samples, const size_t parameters, const size_t index,
                cudaStream_t stream)
{
    // one thread for one sample
    int blockSize = NTHREADS;
    int gridSize = IDIVUP(samples, blockSize);
    cudaLangevin_kernels::updateTheta_kernel<realtype_t><<<gridSize, blockSize, 0, stream>>>(
        theta, prior_gradient, datalikelihood_gradient, half_epsilon_t, eta_t, 
        samples, parameters, index);
    cudaCheckError("cudaLangevin:updateTheta Error");
}

// explicit instantiation
template void altar::cuda::bayesian::cudaLangevin::
    updateTheta<float>(float * const,
                const float * const,
                const float * const,
                const float, const float * const,
                const size_t, const size_t, const size_t,
                cudaStream_t);
template void altar::cuda::bayesian::cudaLangevin::
    updateTheta<double>(double * const,
                const double * const,
                const double * const,
                const double, const double * const,
                const size_t, const size_t, const size_t,
                cudaStream_t);


template <typename realtype_t>
void altar::cuda::bayesian::cudaLangevin::
updateThetaBatched(realtype_t * const theta,
                const realtype_t alpha1, const realtype_t * const prior_gradient,
                const realtype_t alpha2, const realtype_t * const datalikelihood_gradient,
                const realtype_t half_epsilon_t, const realtype_t * const eta_t,
                const size_t samples, const size_t parameters,
                cudaStream_t stream)
{
    // one thread for one sample
    int blockSize = NTHREADS;
    int gridSize = IDIVUP(samples, blockSize);
    cudaLangevin_kernels::updateThetaBatched_kernel<realtype_t><<<gridSize, blockSize, 0, stream>>>(
        theta, alpha1, prior_gradient, alpha2, datalikelihood_gradient, half_epsilon_t, eta_t,
        samples, parameters);
    cudaCheckError("cudaLangevin:updateThetaBatched Error");
}

// explicit instantiation
template void altar::cuda::bayesian::cudaLangevin::
    updateThetaBatched<float>(float * const,
                const float, const float * const,
                const float, const float * const,
                const float, const float * const,
                const size_t, const size_t,
                cudaStream_t);
template void altar::cuda::bayesian::cudaLangevin::
    updateThetaBatched<double>(double * const,
                const double, const double * const,
                const double, const double * const,
                const double, const double * const,
                const size_t, const size_t,
                cudaStream_t);


// sgld update theta, one parameter only
template <typename realtype_t>
__global__ void
cudaLangevin_kernels::
updateTheta_kernel(realtype_t * const theta,
                const realtype_t * const prior_gradient,
                const realtype_t * const datalikelihood_gradient,
                const realtype_t half_epsilon_t, const realtype_t * const eta_t,
                const size_t samples, const size_t parameters, const size_t index)
{
    int sample = blockIdx.x*blockDim.x + threadIdx.x;
    if (sample >= samples) return;

    theta[sample*parameters+index] +=
        half_epsilon_t*(prior_gradient[sample]+datalikelihood_gradient[sample])
        + eta_t[sample];
    // all done
}

// sgld update theta, all parameters
template <typename realtype_t>
__global__ void
cudaLangevin_kernels::
updateThetaBatched_kernel(realtype_t * const theta,
                const realtype_t alpha1, const realtype_t * const prior_gradient,
                const realtype_t alpha2, const realtype_t * const datalikelihood_gradient,
                const realtype_t half_epsilon_t, const realtype_t * const eta_t,
                const size_t samples, const size_t parameters)
{
    int sample = blockIdx.x*blockDim.x + threadIdx.x;
    if (sample >= samples) return;

    realtype_t * theta_sample = theta + sample*parameters;
    const realtype_t * prior_gradient_sample = prior_gradient + sample*parameters;
    const realtype_t * datalikelihood_gradient_sample = datalikelihood_gradient + sample*parameters;
    const realtype_t * eta_t_sample = eta_t + sample*parameters;

    for(int parameter=0; parameter<parameters; parameter++)
        theta_sample[parameter] +=
            half_epsilon_t * (alpha1*prior_gradient_sample[parameter]
                + alpha2*datalikelihood_gradient_sample[parameter])
            + eta_t_sample[parameter];
    // all done
}

// end of file
