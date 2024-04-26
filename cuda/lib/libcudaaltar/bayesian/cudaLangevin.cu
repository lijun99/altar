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

}


/// @brief queue valid samples to a new theta
/// @param samples: batch or valid samples
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

// metropolis acceptance/rejection
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

// end of file
