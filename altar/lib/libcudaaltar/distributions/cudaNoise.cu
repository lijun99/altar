// -*- C++ -*-
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//

#include "cudaNoise.h"
#include "../support.h"

#include <pyre/cuda.h>

namespace cudaNoise_kernels {

    // one thread per sample
    template <typename real_type>
    __global__ void
    _likelihood(matrix_view_t<real_type> theta, vector_view_t<real_type> likelihood,
        const size_t column, const real_type observations, const real_type normalization,
        const size_t batch)
    {
        size_t sample = blockIdx.x*blockDim.x + threadIdx.x;
        if (sample >= batch) return;
        int s = static_cast<int>(sample);
        auto h = theta[{ s, static_cast<int>(column) }];
        auto llk = likelihood[{ s }];
        likelihood[{ s }] = normalization - observations * h + exp(-2 * h) * (llk - normalization);
    }

    // one thread per sample and column of {data_gradient}
    template <typename real_type>
    __global__ void
    _gradient(matrix_view_t<real_type> theta, vector_view_t<real_type, true> likelihood,
        matrix_view_t<real_type, false> data_gradient, matrix_view_t<real_type, false> gradient,
        const size_t column, const real_type observations, const real_type normalization,
        const size_t batch)
    {
        auto parameters = data_gradient.packing().shape()[1];
        size_t index = blockIdx.x*blockDim.x + threadIdx.x;
        if (index >= batch*parameters) return;
        int sample = index / parameters;
        int j = index % parameters;
        auto scale = exp(-2 * theta[{ sample, static_cast<int>(column) }]);
        data_gradient[{ sample, j }] *= scale;
        if (j == 0) {
            // -N + exp(-2h) chi^2, with chi^2 = 2 (c - llk)
            gradient[{ sample, static_cast<int>(column) }] +=
                -observations + 2 * scale * (normalization - likelihood[{ sample }]);
        }
    }

} // of namespace cudaNoise_kernels

template <typename real_type>
void altar::cuda::distributions::cudaNoise::
likelihood(matrix_view_t<real_type> theta, vector_view_t<real_type> likelihood,
    const size_t column, const real_type observations, const real_type normalization,
    const size_t batch, cudaStream_t stream)
{
    if (batch == 0) return;
    auto blockSize = NTHREADS;
    auto gridSize = IDIVUP(batch, blockSize);
    cudaNoise_kernels::_likelihood<real_type><<<gridSize, blockSize, 0, stream>>>(
        theta, likelihood, column, observations, normalization, batch);
    cudaCheckError("cudaNoise::likelihood error");
}

template void altar::cuda::distributions::cudaNoise::likelihood<float>(
    matrix_view_t<float>, vector_view_t<float>, const size_t, const float, const float,
    const size_t, cudaStream_t);
template void altar::cuda::distributions::cudaNoise::likelihood<double>(
    matrix_view_t<double>, vector_view_t<double>, const size_t, const double, const double,
    const size_t, cudaStream_t);


template <typename real_type>
void altar::cuda::distributions::cudaNoise::
gradient(matrix_view_t<real_type> theta, vector_view_t<real_type, true> likelihood,
    matrix_view_t<real_type, false> data_gradient, matrix_view_t<real_type, false> gradient,
    const size_t column, const real_type observations, const real_type normalization,
    const size_t batch, cudaStream_t stream)
{
    auto cells = batch * data_gradient.packing().shape()[1];
    if (cells == 0) return;
    auto blockSize = NTHREADS;
    auto gridSize = IDIVUP(cells, blockSize);
    cudaNoise_kernels::_gradient<real_type><<<gridSize, blockSize, 0, stream>>>(
        theta, likelihood, data_gradient, gradient, column, observations, normalization, batch);
    cudaCheckError("cudaNoise::gradient error");
}

template void altar::cuda::distributions::cudaNoise::gradient<float>(
    matrix_view_t<float>, vector_view_t<float, true>, matrix_view_t<float, false>,
    matrix_view_t<float, false>, const size_t, const float, const float, const size_t, cudaStream_t);
template void altar::cuda::distributions::cudaNoise::gradient<double>(
    matrix_view_t<double>, vector_view_t<double, true>, matrix_view_t<double, false>,
    matrix_view_t<double, false>, const size_t, const double, const double, const size_t, cudaStream_t);

// end of file
