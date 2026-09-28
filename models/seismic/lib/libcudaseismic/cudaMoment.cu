// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//

// declarations
#include "cudaMoment.h"

// cuda kernels; defined ahead of the launchers that instantiate them
namespace cudaMoment_kernels {

    // the sample's M0 = sum_i mu_area[i] theta[sample, idx_begin+i]
    template <typename real_type>
    __device__ real_type
    _moment(matrix_view_t<real_type> theta, int sample,
        const size_t idx_begin, const size_t idx_end, vector_view_t<real_type, true> mu_area)
    {
        auto M0 = real_type{0};
        for (auto i = idx_begin; i < idx_end; ++i) {
            M0 += mu_area[{ static_cast<int>(i - idx_begin) }] * theta[{ sample, static_cast<int>(i) }];
        }
        return M0;
    }

    // one thread per sample: likelihood[sample] -= factor*(Mw - mean)^2/(2 sigma^2)
    template <typename real_type>
    __global__ void
    _logpdf(matrix_view_t<real_type> theta, vector_view_t<real_type> likelihood,
        const size_t idx_begin, const size_t idx_end,
        const real_type mean, const real_type sigma,
        vector_view_t<real_type, true> mu_area, const real_type factor)
    {
        int sample = blockIdx.x*blockDim.x + threadIdx.x;
        auto samples = theta.packing().shape()[0];
        if (sample >= samples) return;

        auto M0 = _moment(theta, sample, idx_begin, idx_end, mu_area);
        auto dMw = (log10(abs(M0)) + real_type(5.9)) / real_type(1.5) - mean;
        likelihood[{ sample }] -= factor * dMw * dMw / (2 * sigma * sigma);
    }

    // one thread per sample: gradient[i] = -factor*(Mw - mean)/sigma^2 * mu_area[i]/(1.5 ln10 M0)
    template <typename real_type>
    __global__ void
    _logpdf_gradient(matrix_view_t<real_type> theta, matrix_view_t<real_type, false> gradient,
        const size_t idx_begin, const size_t idx_end,
        const real_type mean, const real_type sigma,
        vector_view_t<real_type, true> mu_area, const real_type factor)
    {
        int sample = blockIdx.x*blockDim.x + threadIdx.x;
        auto samples = theta.packing().shape()[0];
        if (sample >= samples) return;

        auto M0 = _moment(theta, sample, idx_begin, idx_end, mu_area);
        auto dMw = (log10(abs(M0)) + real_type(5.9)) / real_type(1.5) - mean;
        auto scale = -factor * dMw / (sigma * sigma) / (real_type(1.5) * log(real_type(10)) * M0);
        for (auto i = idx_begin; i < idx_end; ++i) {
            gradient[{ sample, static_cast<int>(i) }] = scale * mu_area[{ static_cast<int>(i - idx_begin) }];
        }
    }

} // of namespace cudaMoment_kernels


// launch {cudaMoment_kernels::_logpdf}
template <typename real_type>
void altar::models::seismic::cudaMoment::
logpdf(matrix_view_t<real_type> theta, vector_view_t<real_type> likelihood,
    const size_t idx_begin, const size_t idx_end,
    const real_type mean, const real_type sigma,
    vector_view_t<real_type, true> mu_area, const real_type factor,
    cudaStream_t stream)
{
    auto samples = theta.packing().shape()[0];
    auto blockSize = NTHREADS;
    auto gridSize = IDIVUP(samples, blockSize);
    cudaMoment_kernels::_logpdf<real_type><<<gridSize, blockSize, 0, stream>>>(
        theta, likelihood, idx_begin, idx_end, mean, sigma, mu_area, factor);
    cudaCheckError("cudaMoment::logpdf error");
}

template void altar::models::seismic::cudaMoment::logpdf<float>(
    matrix_view_t<float>, vector_view_t<float>, const size_t, const size_t,
    const float, const float, vector_view_t<float, true>, const float, cudaStream_t);
template void altar::models::seismic::cudaMoment::logpdf<double>(
    matrix_view_t<double>, vector_view_t<double>, const size_t, const size_t,
    const double, const double, vector_view_t<double, true>, const double, cudaStream_t);

// launch {cudaMoment_kernels::_logpdf_gradient}
template <typename real_type>
void altar::models::seismic::cudaMoment::
logpdf_gradient(matrix_view_t<real_type> theta, matrix_view_t<real_type, false> gradient,
    const size_t idx_begin, const size_t idx_end,
    const real_type mean, const real_type sigma,
    vector_view_t<real_type, true> mu_area, const real_type factor,
    cudaStream_t stream)
{
    auto samples = theta.packing().shape()[0];
    auto blockSize = NTHREADS;
    auto gridSize = IDIVUP(samples, blockSize);
    cudaMoment_kernels::_logpdf_gradient<real_type><<<gridSize, blockSize, 0, stream>>>(
        theta, gradient, idx_begin, idx_end, mean, sigma, mu_area, factor);
    cudaCheckError("cudaMoment::logpdf_gradient error");
}

template void altar::models::seismic::cudaMoment::logpdf_gradient<float>(
    matrix_view_t<float>, matrix_view_t<float, false>, const size_t, const size_t,
    const float, const float, vector_view_t<float, true>, const float, cudaStream_t);
template void altar::models::seismic::cudaMoment::logpdf_gradient<double>(
    matrix_view_t<double>, matrix_view_t<double, false>, const size_t, const size_t,
    const double, const double, vector_view_t<double, true>, const double, cudaStream_t);

// end of file
