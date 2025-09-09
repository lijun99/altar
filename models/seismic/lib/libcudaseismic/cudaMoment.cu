// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-2021 parasim inc
// (c) 2010-2021 california institute of technology
// all rights reserved
//
// Author(s): Hailiang Zhang, Lijun Zhu


// declarations
#include "cudaMoment.h"

// cuda utilities
#include <pyre/cuda.h>
#include <curand_kernel.h>

// cuda kernel declarations
namespace cudaMoment_kernels {
    // log pdf
    template<typename real_type>
    __global__ void _logpdf(const real_type * const theta, real_type * const probability,
        const size_t samples, const size_t parameters,
        const size_t idx_begin, const size_t idx_end,
        const real_type mean, const real_type sigma,
        const real_type * const mu_area, const real_type moment_constraint_factor);
}



// compute log probability
template <typename real_type>
void altar::models::seismic::cudaMoment::
logpdf(const real_type * const theta, real_type * const probability,
                    const size_t samples, const size_t parameters,
                    const size_t idx_begin, const size_t idx_end,
                    const real_type mean, const real_type sigma,
                    const real_type * const mu_area, const real_type moment_constraint_factor,
                    cudaStream_t stream)
{
    int blockSize = NTHREADS;
    int gridSize = IDIVUP(samples, blockSize);

    // call cuda kernels
    cudaMoment_kernels::_logpdf<real_type><<<gridSize, blockSize, 0, stream>>>(
        theta, probability, samples, parameters, idx_begin, idx_end, mean, sigma, mu_area, moment_constraint_factor);
    cudaCheckError("cudaMoment:: log_pdf error");
}

// explicit instantiation
template void altar::models::seismic::cudaMoment::logpdf<float>(
    const float * const theta, float * const probability,
    const size_t samples, const size_t parameters,
    const size_t idx_begin, const size_t idx_end,
    const float mean, const float sigma,
    const float * const mu_area,
    const float moment_constraint_factor,
    cudaStream_t stream);

template void altar::models::seismic::cudaMoment::logpdf<double>(
    const double * const theta, double * const probability,
    const size_t samples, const size_t parameters,
    const size_t idx_begin, const size_t idx_end,
    const double mean, const double sigma,
    const double * const mu_area,
    const double moment_constraint_factor,
    cudaStream_t stream);

namespace cudaMoment_kernels {

//log_pdf kernel
template <typename real_type>
__global__ void
_logpdf(const real_type * const theta, real_type * const probability,
        const size_t samples, const size_t parameters,
        const size_t idx_begin, const size_t idx_end,
        const real_type mean, const real_type sigma,
        const real_type * const mu_area, const real_type moment_constraint_factor)
{
    // get the thread/sample id
    int sample = blockIdx.x*blockDim.x + threadIdx.x;
    if (sample >= samples) return;

    // get the theta pointer
    const real_type * theta_sample = theta + sample*parameters;

    // neglect the constant piece to be compared with no contrain
    // real_type c1 = -log( sigma * sqrt(2.*PI) );
    real_type c2 = 0.5/(sigma*sigma);

    // compute the moment M_0 by summing (mu A)_i D_i
    real_type M0 = 0.0;
    for (int i=idx_begin; i<idx_end; ++i)
    {       
        M0 += mu_area[i-idx_begin]*theta_sample[i];
    }

    // Moment magnitude w mu in GPa, A in km^2, D in m
    real_type Mw = (log10(abs(M0)) + 5.9)/1.5; 

    // logpdf = -c2 * (|M0| - mean)^2, neglecting constant
    Mw -= mean;
    probability[sample] -= moment_constraint_factor*c2*Mw*Mw;
}

} // of namespace cudaMoment_kernels

// end of file
