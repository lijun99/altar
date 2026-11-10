// -*- C++ -*-
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//

#include "cudaPolar.h"
#include "../support.h"

#include <pyre/cuda.h>

namespace cudaPolar_kernels {

    // one thread per sample and column of {extended}
    template <typename real_type>
    __global__ void
    _extend(matrix_view_t<real_type> theta, matrix_view_t<real_type, false> extended,
        const size_t magnitude, const size_t rake, const size_t count)
    {
        auto rows = extended.packing().shape()[0];
        auto width = extended.packing().shape()[1];
        auto parameters = theta.packing().shape()[1];
        size_t index = blockIdx.x*blockDim.x + threadIdx.x;
        if (index >= rows*width) return;
        int sample = index / width;
        int column = index % width;
        if (column < parameters) {
            extended[{ sample, column }] = theta[{ sample, column }];
            return;
        }
        int k = (column - parameters) % count;
        auto m = theta[{ sample, static_cast<int>(magnitude + k) }];
        auto r = theta[{ sample, static_cast<int>(rake + k) }] * real_type(M_PI / 180);
        extended[{ sample, column }] = column - parameters < count ? m * cos(r) : m * sin(r);
    }

    // one thread per sample and column of {gradient}
    template <typename real_type>
    __global__ void
    _fold(matrix_view_t<real_type> theta, matrix_view_t<real_type> gradient_extended,
        matrix_view_t<real_type, false> gradient,
        const size_t magnitude, const size_t rake, const size_t count)
    {
        auto rows = gradient.packing().shape()[0];
        auto parameters = gradient.packing().shape()[1];
        size_t index = blockIdx.x*blockDim.x + threadIdx.x;
        if (index >= rows*parameters) return;
        int sample = index / parameters;
        int column = index % parameters;
        auto g = gradient_extended[{ sample, column }];
        if (column >= magnitude && column < magnitude + count) {
            // d/dm = cos(r) g_along + sin(r) g_across
            int k = column - magnitude;
            auto r = theta[{ sample, static_cast<int>(rake + k) }] * real_type(M_PI / 180);
            auto along = gradient_extended[{ sample, static_cast<int>(parameters + k) }];
            auto across = gradient_extended[{ sample, static_cast<int>(parameters + count + k) }];
            g += cos(r) * along + sin(r) * across;
        } else if (column >= rake && column < rake + count) {
            // d/drake = m (cos(r) g_across - sin(r) g_along) pi/180
            int k = column - rake;
            auto m = theta[{ sample, static_cast<int>(magnitude + k) }];
            auto r = theta[{ sample, column }] * real_type(M_PI / 180);
            auto along = gradient_extended[{ sample, static_cast<int>(parameters + k) }];
            auto across = gradient_extended[{ sample, static_cast<int>(parameters + count + k) }];
            g += m * (cos(r) * across - sin(r) * along) * real_type(M_PI / 180);
        }
        gradient[{ sample, column }] = g;
    }

} // of namespace cudaPolar_kernels

template <typename real_type>
void altar::cuda::distributions::cudaPolar::
extend(matrix_view_t<real_type> theta, matrix_view_t<real_type, false> extended,
    const size_t magnitude, const size_t rake, const size_t count, cudaStream_t stream)
{
    auto cells = extended.packing().shape()[0] * extended.packing().shape()[1];
    auto blockSize = NTHREADS;
    auto gridSize = IDIVUP(cells, blockSize);
    cudaPolar_kernels::_extend<real_type><<<gridSize, blockSize, 0, stream>>>(
        theta, extended, magnitude, rake, count);
    cudaCheckError("cudaPolar::extend error");
}

template void altar::cuda::distributions::cudaPolar::extend<float>(
    matrix_view_t<float>, matrix_view_t<float, false>, const size_t, const size_t, const size_t, cudaStream_t);
template void altar::cuda::distributions::cudaPolar::extend<double>(
    matrix_view_t<double>, matrix_view_t<double, false>, const size_t, const size_t, const size_t, cudaStream_t);


template <typename real_type>
void altar::cuda::distributions::cudaPolar::
fold(matrix_view_t<real_type> theta, matrix_view_t<real_type> gradient_extended,
    matrix_view_t<real_type, false> gradient,
    const size_t magnitude, const size_t rake, const size_t count, cudaStream_t stream)
{
    auto cells = gradient.packing().shape()[0] * gradient.packing().shape()[1];
    auto blockSize = NTHREADS;
    auto gridSize = IDIVUP(cells, blockSize);
    cudaPolar_kernels::_fold<real_type><<<gridSize, blockSize, 0, stream>>>(
        theta, gradient_extended, gradient, magnitude, rake, count);
    cudaCheckError("cudaPolar::fold error");
}

template void altar::cuda::distributions::cudaPolar::fold<float>(
    matrix_view_t<float>, matrix_view_t<float>, matrix_view_t<float, false>,
    const size_t, const size_t, const size_t, cudaStream_t);
template void altar::cuda::distributions::cudaPolar::fold<double>(
    matrix_view_t<double>, matrix_view_t<double>, matrix_view_t<double, false>,
    const size_t, const size_t, const size_t, cudaStream_t);

// end of file
