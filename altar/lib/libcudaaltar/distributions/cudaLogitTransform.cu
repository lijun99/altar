// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//

// declarations
#include "cudaLogitTransform.h"

// cuda kernels; defined here, ahead of the launcher functions below that instantiate and
// launch them
namespace cudaLogitTransform_kernels {

    // one thread per sample: theta[idx_begin:idx_end] <- low + (high-low)*sigmoid(theta)
    template <typename real_type>
    __global__ void
    _to_physical(matrix_view_t<real_type, false> theta,
        const size_t idx_begin, const size_t idx_end,
        const real_type low, const real_type high)
    {
        int sample = blockIdx.x*blockDim.x + threadIdx.x;
        auto samples = theta.packing().shape()[0];
        if (sample >= samples) return;

        auto range = high - low;
        for (auto i = idx_begin; i < idx_end; ++i) {
            auto & v = theta[{ sample, static_cast<int>(i) }];
            v = low + range / (1 + exp(-v));
        }
    }

    // one thread per sample: theta[idx_begin:idx_end] <- logit((theta-low)/(high-low))
    template <typename real_type>
    __global__ void
    _to_sampling(matrix_view_t<real_type, false> theta,
        const size_t idx_begin, const size_t idx_end,
        const real_type low, const real_type high)
    {
        int sample = blockIdx.x*blockDim.x + threadIdx.x;
        auto samples = theta.packing().shape()[0];
        if (sample >= samples) return;

        auto range = high - low;
        for (auto i = idx_begin; i < idx_end; ++i) {
            auto & v = theta[{ sample, static_cast<int>(i) }];
            auto u = (v - low) / range;
            v = log(u / (1 - u));
        }
    }

    // one thread per sample: jacobian[idx_begin:idx_end] <- (high-low)*sig*(1-sig)
    template <typename real_type>
    __global__ void
    _jacobian(matrix_view_t<real_type> theta, matrix_view_t<real_type, false> jacobian,
        const size_t idx_begin, const size_t idx_end,
        const real_type low, const real_type high)
    {
        int sample = blockIdx.x*blockDim.x + threadIdx.x;
        auto samples = theta.packing().shape()[0];
        if (sample >= samples) return;

        auto range = high - low;
        for (auto i = idx_begin; i < idx_end; ++i) {
            auto sig = (theta[{ sample, static_cast<int>(i) }] - low) / range;
            jacobian[{ sample, static_cast<int>(i) }] = range * sig * (1 - sig);
        }
    }

    // one thread per sample: gradient[idx_begin:idx_end] <- 1 - 2*sig
    template <typename real_type>
    __global__ void
    _jacobian_gradient(matrix_view_t<real_type> theta, matrix_view_t<real_type, false> gradient,
        const size_t idx_begin, const size_t idx_end,
        const real_type low, const real_type high)
    {
        int sample = blockIdx.x*blockDim.x + threadIdx.x;
        auto samples = theta.packing().shape()[0];
        if (sample >= samples) return;

        auto range = high - low;
        for (auto i = idx_begin; i < idx_end; ++i) {
            auto sig = (theta[{ sample, static_cast<int>(i) }] - low) / range;
            gradient[{ sample, static_cast<int>(i) }] = 1 - 2*sig;
        }
    }

    // one thread per sample: likelihood[sample] += sum_i log(sig) + log(1-sig)
    template <typename real_type>
    __global__ void
    _log_jacobian(matrix_view_t<real_type> theta, vector_view_t<real_type> likelihood,
        const size_t idx_begin, const size_t idx_end,
        const real_type low, const real_type high)
    {
        int sample = blockIdx.x*blockDim.x + threadIdx.x;
        auto samples = theta.packing().shape()[0];
        if (sample >= samples) return;

        auto range = high - low;
        auto contribution = real_type{0};
        for (auto i = idx_begin; i < idx_end; ++i) {
            auto sig = (theta[{ sample, static_cast<int>(i) }] - low) / range;
            contribution += log(sig) + log(1 - sig);
        }
        likelihood[{ sample }] += contribution;
    }

} // of namespace cudaLogitTransform_kernels


// launch {cudaLogitTransform_kernels::_to_physical}
template <typename real_type>
void altar::cuda::distributions::cudaLogitTransform::
to_physical(matrix_view_t<real_type, false> theta,
    const size_t idx_begin, const size_t idx_end,
    const real_type low, const real_type high,
    cudaStream_t stream)
{
    auto samples = theta.packing().shape()[0];
    auto blockSize = NTHREADS;
    auto gridSize = IDIVUP(samples, blockSize);
    cudaLogitTransform_kernels::_to_physical<real_type><<<gridSize, blockSize, 0, stream>>>(
        theta, idx_begin, idx_end, low, high);
    cudaCheckError("cudaLogitTransform::to_physical error");
}

template void altar::cuda::distributions::cudaLogitTransform::to_physical<float>(
    matrix_view_t<float, false>, const size_t, const size_t, const float, const float, cudaStream_t);
template void altar::cuda::distributions::cudaLogitTransform::to_physical<double>(
    matrix_view_t<double, false>, const size_t, const size_t, const double, const double, cudaStream_t);


// launch {cudaLogitTransform_kernels::_to_sampling}
template <typename real_type>
void altar::cuda::distributions::cudaLogitTransform::
to_sampling(matrix_view_t<real_type, false> theta,
    const size_t idx_begin, const size_t idx_end,
    const real_type low, const real_type high,
    cudaStream_t stream)
{
    auto samples = theta.packing().shape()[0];
    auto blockSize = NTHREADS;
    auto gridSize = IDIVUP(samples, blockSize);
    cudaLogitTransform_kernels::_to_sampling<real_type><<<gridSize, blockSize, 0, stream>>>(
        theta, idx_begin, idx_end, low, high);
    cudaCheckError("cudaLogitTransform::to_sampling error");
}

template void altar::cuda::distributions::cudaLogitTransform::to_sampling<float>(
    matrix_view_t<float, false>, const size_t, const size_t, const float, const float, cudaStream_t);
template void altar::cuda::distributions::cudaLogitTransform::to_sampling<double>(
    matrix_view_t<double, false>, const size_t, const size_t, const double, const double, cudaStream_t);


// launch {cudaLogitTransform_kernels::_jacobian}
template <typename real_type>
void altar::cuda::distributions::cudaLogitTransform::
jacobian(matrix_view_t<real_type> theta, matrix_view_t<real_type, false> jacobian,
    const size_t idx_begin, const size_t idx_end,
    const real_type low, const real_type high,
    cudaStream_t stream)
{
    auto samples = theta.packing().shape()[0];
    auto blockSize = NTHREADS;
    auto gridSize = IDIVUP(samples, blockSize);
    cudaLogitTransform_kernels::_jacobian<real_type><<<gridSize, blockSize, 0, stream>>>(
        theta, jacobian, idx_begin, idx_end, low, high);
    cudaCheckError("cudaLogitTransform::jacobian error");
}

template void altar::cuda::distributions::cudaLogitTransform::jacobian<float>(
    matrix_view_t<float>, matrix_view_t<float, false>, const size_t, const size_t, const float, const float, cudaStream_t);
template void altar::cuda::distributions::cudaLogitTransform::jacobian<double>(
    matrix_view_t<double>, matrix_view_t<double, false>, const size_t, const size_t, const double, const double, cudaStream_t);


// launch {cudaLogitTransform_kernels::_jacobian_gradient}
template <typename real_type>
void altar::cuda::distributions::cudaLogitTransform::
jacobian_gradient(matrix_view_t<real_type> theta, matrix_view_t<real_type, false> gradient,
    const size_t idx_begin, const size_t idx_end,
    const real_type low, const real_type high,
    cudaStream_t stream)
{
    auto samples = theta.packing().shape()[0];
    auto blockSize = NTHREADS;
    auto gridSize = IDIVUP(samples, blockSize);
    cudaLogitTransform_kernels::_jacobian_gradient<real_type><<<gridSize, blockSize, 0, stream>>>(
        theta, gradient, idx_begin, idx_end, low, high);
    cudaCheckError("cudaLogitTransform::jacobian_gradient error");
}

template void altar::cuda::distributions::cudaLogitTransform::jacobian_gradient<float>(
    matrix_view_t<float>, matrix_view_t<float, false>, const size_t, const size_t, const float, const float, cudaStream_t);
template void altar::cuda::distributions::cudaLogitTransform::jacobian_gradient<double>(
    matrix_view_t<double>, matrix_view_t<double, false>, const size_t, const size_t, const double, const double, cudaStream_t);


// launch {cudaLogitTransform_kernels::_log_jacobian}
template <typename real_type>
void altar::cuda::distributions::cudaLogitTransform::
log_jacobian(matrix_view_t<real_type> theta, vector_view_t<real_type> likelihood,
    const size_t idx_begin, const size_t idx_end,
    const real_type low, const real_type high,
    cudaStream_t stream)
{
    auto samples = theta.packing().shape()[0];
    auto blockSize = NTHREADS;
    auto gridSize = IDIVUP(samples, blockSize);
    cudaLogitTransform_kernels::_log_jacobian<real_type><<<gridSize, blockSize, 0, stream>>>(
        theta, likelihood, idx_begin, idx_end, low, high);
    cudaCheckError("cudaLogitTransform::log_jacobian error");
}

template void altar::cuda::distributions::cudaLogitTransform::log_jacobian<float>(
    matrix_view_t<float>, vector_view_t<float>, const size_t, const size_t, const float, const float, cudaStream_t);
template void altar::cuda::distributions::cudaLogitTransform::log_jacobian<double>(
    matrix_view_t<double>, vector_view_t<double>, const size_t, const size_t, const double, const double, cudaStream_t);

// end of file
