// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-2025 parasim inc
// (c) 2010-2025 california institute of technology
// all rights reserved
//
// Author(s):  Lijun Zhu


// declarations
#include "cudaUniformLogit.h"
// dependencies
#include "cudaRandom.h"
#include "cudaUniform.h"
// cuda utilities
#include <pyre/cuda.h>
#include <curand_kernel.h>

// cuda kernel declarations
namespace cudaUniformLogit_kernels {


    template<typename real_type>
    __global__ void _tosampling(const real_type * const theta,
        real_type * const phi,
        const size_t samples, const size_t parameters,
        const size_t idx_begin, const size_t idx_end,
        const real_type low, const real_type high);

    template<typename real_type>
    __global__ void _tophysical(const real_type * const phi,
        real_type * const theta,
        const size_t samples, const size_t parameters,
        const size_t idx_begin, const size_t idx_end,
        const real_type low, const real_type high);

    template<typename real_type>
    __global__ void _logpdf(const real_type * const theta, real_type * const probability,
        const size_t samples, const size_t parameters,
        const size_t idx_begin, const size_t idx_end,
        const real_type low, const real_type high);

    template <typename real_type>
    __global__ void _logpdfgradient(const real_type * const theta, real_type * const gradient,
        const size_t samples, const size_t parameters,
        const size_t idx_begin, const size_t idx_end,
        const real_type low, const real_type high);

    template<typename real_type>
    __global__ void _jacobian(const real_type * const theta,
        real_type * const jacobian,
        const size_t samples, const size_t parameters,
        const size_t idx_begin, const size_t idx_end,
        const real_type low, const real_type high);

} // of namespace cudaUniformLogit_kernels



// transform from sampling to physical space
template <typename real_type>
void altar::cuda::distributions::cudaUniformLogit::
tophysical(const real_type * const phi, real_type * const theta,
    const size_t samples, const size_t parameters,
    const size_t idx_begin, const size_t idx_end,
    const real_type low, const real_type high,
    cudaStream_t stream)
{
    int blockSize = NTHREADS;
    int gridSize = IDIVUP(samples, blockSize);
    // call cuda kernels
    cudaUniformLogit_kernels::_tophysical<real_type><<<gridSize, blockSize, 0, stream>>>(
        phi, theta, samples, parameters, idx_begin, idx_end, low, high);
    cudaCheckError("cudaUniformLogit:: tophysical error");
}

// explicit instantiation
template void altar::cuda::distributions::cudaUniformLogit::tophysical<float>(
    const float * const, float * const,
    const size_t, const size_t,
    const size_t, const size_t,
    const float, const float, cudaStream_t);
template void altar::cuda::distributions::cudaUniformLogit::tophysical<double>(
    const double * const, double * const,
    const size_t, const size_t,
    const size_t, const size_t,
    const double, const double, cudaStream_t);

// transform logistic variable to bounded
template <typename real_type>
void altar::cuda::distributions::cudaUniformLogit::
tosampling(const real_type * const theta, real_type * const phi,
        const size_t samples, const size_t parameters,
        const size_t idx_begin, const size_t idx_end,
        const real_type low, const real_type high,
        cudaStream_t stream)
{
    int blockSize = NTHREADS;
    int gridSize = IDIVUP(samples, blockSize);
    // call cuda kernels
    cudaUniformLogit_kernels::_tosampling<real_type><<<gridSize, blockSize, 0, stream>>>(
        theta, phi, samples, parameters, idx_begin, idx_end, low, high);
    cudaCheckError("cudaUniformLogit:: tosampling error");
}

// explicit instantiation
template void altar::cuda::distributions::cudaUniformLogit::tosampling<float>(
    const float * const, float * const,
    const size_t, const size_t,
    const size_t, const size_t,
    const float, const float, cudaStream_t);
template void altar::cuda::distributions::cudaUniformLogit::tosampling<double>(
    const double * const, double * const,
    const size_t, const size_t,
    const size_t, const size_t,
    const double, const double, cudaStream_t);

//! compute jacobian matrix dtheta/dphi
//! @note use theta as inputs for efficiency
template <typename real_type>
void altar::cuda::distributions::cudaUniformLogit::
jacobian(const real_type * const theta,
    real_type * const jacobian,
    const size_t samples, const size_t parameters,
    const size_t idx_begin, const size_t idx_end,
    const real_type low, const real_type high,
    cudaStream_t stream)
{
    int blockSize = NTHREADS;
    int gridSize = IDIVUP(samples, blockSize);
    // call cuda kernels
    cudaUniformLogit_kernels::_jacobian<real_type><<<gridSize, blockSize, 0, stream>>>(
        theta, jacobian, samples, parameters,
        idx_begin, idx_end, low, high);
    cudaCheckError("cudaUniformLogit:: jacobian error");
}

// explicit instantiation
template void altar::cuda::distributions::cudaUniformLogit::jacobian<float>(
    const float * const, float * const,
    const size_t, const size_t, const size_t, const size_t,
    const float, const float, cudaStream_t);
template void altar::cuda::distributions::cudaUniformLogit::jacobian<double>(
    const double * const, double * const,
    const size_t, const size_t, const size_t, const size_t,
    const double, const double, cudaStream_t);

// compute log pdf
template <typename real_type>
void altar::cuda::distributions::cudaUniformLogit::
logpdf(const real_type * const theta,
    real_type * const probability,
    const size_t samples, const size_t parameters,
    const size_t idx_begin, const size_t idx_end,
    const real_type low, const real_type high,
    cudaStream_t stream)
{
    int blockSize = NTHREADS;
    int gridSize = IDIVUP(samples, blockSize);
    // call cuda kernels
    cudaUniformLogit_kernels::_logpdf<real_type><<<gridSize, blockSize, 0, stream>>>(
        theta, probability, samples, parameters,
        idx_begin, idx_end, low, high);
    cudaCheckError("cudaUniformLogit:: logpdf error");
}

// explicit instantiation
template void altar::cuda::distributions::cudaUniformLogit::logpdf<float>(
    const float * const, float * const,
    const size_t, const size_t, const size_t, const size_t,
    const float, const float, cudaStream_t);
template void altar::cuda::distributions::cudaUniformLogit::logpdf<double>(
    const double * const, double * const,
    const size_t, const size_t, const size_t, const size_t,
    const double, const double, cudaStream_t);

// compute log pdf gradient
template <typename real_type>
void altar::cuda::distributions::cudaUniformLogit::
logpdfgradient(const real_type * const theta,
    real_type * const gradient,
    const size_t samples, const size_t parameters,
    const size_t idx_begin, const size_t idx_end,
    const real_type low, const real_type high,
    cudaStream_t stream)
{
    int blockSize = NTHREADS;
    int gridSize = IDIVUP(samples, blockSize);
    // call cuda kernels
    cudaUniformLogit_kernels::_logpdfgradient<real_type><<<gridSize, blockSize, 0, stream>>>(
        theta, gradient, samples, parameters,
        idx_begin, idx_end, low, high);
    cudaCheckError("cudaUniformLogit:: logpdfgradient error");
}

// explicit instantiation
template void altar::cuda::distributions::cudaUniformLogit::logpdfgradient<float>(
    const float * const, float * const,
    const size_t, const size_t, const size_t, const size_t,
    const float, const float,
    cudaStream_t);
template void altar::cuda::distributions::cudaUniformLogit::logpdfgradient<double>(
    const double * const, double * const,
    const size_t, const size_t, const size_t, const size_t,
    const double, const double,
    cudaStream_t);


namespace cudaUniformLogit_kernels {

//! tophysical kernel (expit/sigmoid)
template <typename real_type>
__global__ void
_tophysical(const real_type * phi, real_type * const theta,
    const size_t samples, const size_t parameters,
    const size_t idx_begin, const size_t idx_end,
    const real_type low, const real_type high)
{
    // get the thread/sample id
    int sample = blockIdx.x*blockDim.x + threadIdx.x;
    if (sample >= samples) return;
    // get the starting pointer for this sample
    const real_type * phi_sample = phi + sample*parameters;
    real_type * theta_sample = theta + sample*parameters;

    real_type range = high - low;

    // iterate over each parameter
    for (int i=idx_begin; i<idx_end; ++i)
    {
        theta_sample[i] = low + range/(1.0 + exp(-phi_sample[i]));
    }
}

//! tosampling kernel (logit)
template <typename real_type>
__global__ void
_tosampling(const real_type * const theta, real_type * const phi,
    const size_t samples, const size_t parameters,
    const size_t idx_begin, const size_t idx_end,
    const real_type low, const real_type high)
{
    // get the thread/sample id
    int sample = blockIdx.x*blockDim.x + threadIdx.x;
    if (sample >= samples) return;
    // get the starting pointer for this sample
    const real_type * theta_sample = theta + sample*parameters;
    real_type * phi_sample = phi + sample*parameters;

    real_type range_inv = 1.0/(high-low);
    // check each parameter
    for (int i=idx_begin; i<idx_end; ++i)
    {
        real_type u = (theta_sample[i]-low)*range_inv;
        phi_sample[i] = log(u/(1.0-u));
    }
}


//! log_pdf kernel for uniformlogit distribution
//! @note we use theta (instead of phi) as input for efficiency
template <typename real_type>
__global__ void
_logpdf(const real_type * const theta, real_type * const probability,
    const size_t samples, const size_t parameters,
    const size_t idx_begin, const size_t idx_end,
    const real_type low, const real_type high)
{
    // get the thread/sample id
    int sample = blockIdx.x*blockDim.x + threadIdx.x;
    if (sample >= samples) return;

    // get the starting pointer for this sample
    const real_type * theta_sample = theta + sample*parameters;

    // accumulated log_pdf for this sample
    real_type log_pdf = 0.0;

    // range inverse
    real_type range_inv = 1.0/(high-low);

    // iterate over parameters in this dataset and in this sample
    for (int i=idx_begin; i<idx_end; ++i)
    {
        real_type u = (theta_sample[i]-low)*range_inv;
        // log pdf of logistic distribution
        log_pdf += log(u*(1.0-u));
    }

    probability[sample] += log_pdf;
}

//! gradient of log pdf kernel for uniformlogit distribution
//! d\log P(\phi) / d\phi = 1 - 2u, where u = (theta-low)/(high-low)
//! @note we use theta (instead of phi) as input for compute efficiency
template <typename real_type>
__global__ void
_logpdfgradient(const real_type * const theta, real_type * const gradient,
    const size_t samples, const size_t parameters,
    const size_t idx_begin, const size_t idx_end,
    const real_type low, const real_type high)
{
     // get the thread/sample id
    int sample = blockIdx.x*blockDim.x + threadIdx.x;
    if (sample >= samples) return;

    // get the pointers for this sample
    const real_type * theta_sample = theta + sample*parameters;
    real_type * grad_sample = gradient + sample*parameters;

    real_type range_inv = 1.0/(high-low);
    // gradient
    for (int i=idx_begin; i<idx_end; ++i)
    {
        real_type u = (theta_sample[i]-low)*range_inv;
        grad_sample[i] = 1.0-2.0*u;
    }
}

// jacobian kernel - compute d\theta/dpi and log determinant
template <typename real_type>
__global__ void
_jacobian(const real_type * const theta,
    real_type * const jacobian,
    const size_t samples, const size_t parameters,
    const size_t idx_begin, const size_t idx_end,
    const real_type low, const real_type high)
{
    // get the thread/sample id
    int sample = blockIdx.x*blockDim.x + threadIdx.x;
    if (sample >= samples) return;

    // get the starting pointers for this sample
    const real_type * theta_sample = theta + sample*parameters;
    real_type * jacobian_sample = jacobian + sample*parameters;
    real_type range = high - low;

    // compute jacobian d\theta/dpi for each parameter
    for (int i=idx_begin; i<idx_end; ++i)
    {
        // convert theta to u (0,1), then to phi
        real_type u = (theta_sample[i]-low)/range;
        // phi = logit(u) = log(u/(1-u))
        // u = sigmoid(phi)
        // compute derivative J=d\theta/d\phi
        jacobian_sample[i] = range * u * (1.0-u);
    }
}



} // of namespace cudaUniformLogit_kernels

// end of file
