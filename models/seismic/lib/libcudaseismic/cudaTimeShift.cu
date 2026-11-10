// -*- C++ -*-
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//

#include "cudaTimeShift.h"

namespace cudaTimeShift_kernels {

    // the Keys cubic kernel, a = -1/2, and its derivative
    template <typename real_type>
    __device__ real_type
    _keys(real_type x)
    {
        x = abs(x);
        if (x < 1) return (real_type(1.5) * x - real_type(2.5)) * x * x + 1;
        if (x < 2) return ((real_type(-0.5) * x + real_type(2.5)) * x - 4) * x + 2;
        return 0;
    }

    template <typename real_type>
    __device__ real_type
    _keys_derivative(real_type x)
    {
        auto sign = x < 0 ? real_type(-1) : real_type(1);
        x = abs(x);
        if (x < 1) return sign * (real_type(4.5) * x - 5) * x;
        if (x < 2) return sign * ((real_type(-1.5) * x + 5) * x - 4);
        return 0;
    }

    // one thread per sample and observation: out[s, o] = in(i - shift/dt) on the trace of o
    template <typename real_type>
    __global__ void
    _shift(matrix_view_t<real_type> in, matrix_view_t<real_type> theta, matrix_view_t<real_type, false> out,
        const size_t column, const size_t traces, const size_t length, const real_type dt)
    {
        auto rows = in.packing().shape()[0];
        auto observations = in.packing().shape()[1];
        size_t index = blockIdx.x*blockDim.x + threadIdx.x;
        if (index >= rows*observations) return;
        int sample = index / observations;
        int o = index % observations;
        int trace = o / length, i = o % length, start = trace * length;
        auto station = trace / traces;
        auto t = i - theta[{ sample, static_cast<int>(column + station) }] / dt;
        int j0 = static_cast<int>(floor(t));
        auto value = real_type(0);
        for (int j = j0 - 1; j <= j0 + 2; ++j) {
            if (j < 0 || j >= static_cast<int>(length)) continue;
            value += in[{ sample, start + j }] * _keys(t - j);
        }
        out[{ sample, o }] = value;
    }

    // one thread per sample and observation: out[s, o] = sum_i in[s, i] keys(i - shift/dt - j) on the trace of o
    template <typename real_type>
    __global__ void
    _adjoint(matrix_view_t<real_type> in, matrix_view_t<real_type> theta, matrix_view_t<real_type, false> out,
        const size_t column, const size_t traces, const size_t length, const real_type dt)
    {
        auto rows = in.packing().shape()[0];
        auto observations = in.packing().shape()[1];
        size_t index = blockIdx.x*blockDim.x + threadIdx.x;
        if (index >= rows*observations) return;
        int sample = index / observations;
        int o = index % observations;
        int trace = o / length, j = o % length, start = trace * length;
        auto station = trace / traces;
        auto delay = theta[{ sample, static_cast<int>(column + station) }] / dt;
        // the outputs i that read input j: |i - delay - j| < 2
        int i0 = static_cast<int>(floor(j + delay));
        auto value = real_type(0);
        for (int i = i0 - 1; i <= i0 + 2; ++i) {
            if (i < 0 || i >= static_cast<int>(length)) continue;
            value += in[{ sample, start + i }] * _keys(i - delay - j);
        }
        out[{ sample, o }] = value;
    }

    // one thread per sample and station
    template <typename real_type>
    __global__ void
    _gradient(matrix_view_t<real_type> x, matrix_view_t<real_type> w, matrix_view_t<real_type> theta,
        matrix_view_t<real_type, false> gradient,
        const size_t column, const size_t traces, const size_t length, const real_type dt,
        const real_type scale, const size_t stations)
    {
        auto rows = x.packing().shape()[0];
        size_t index = blockIdx.x*blockDim.x + threadIdx.x;
        if (index >= rows*stations) return;
        int sample = index / stations;
        int station = index % stations;
        auto delay = theta[{ sample, static_cast<int>(column + station) }] / dt;
        auto total = real_type(0);
        for (size_t trace = station * traces; trace < (station + 1) * traces; ++trace) {
            int start = trace * length;
            for (int i = 0; i < static_cast<int>(length); ++i) {
                auto t = i - delay;
                int j0 = static_cast<int>(floor(t));
                // d y_i / d shift = -sum_j x_j keys'(t - j) / dt
                auto dy = real_type(0);
                for (int j = j0 - 1; j <= j0 + 2; ++j) {
                    if (j < 0 || j >= static_cast<int>(length)) continue;
                    dy -= x[{ sample, start + j }] * _keys_derivative(t - j);
                }
                total += w[{ sample, start + i }] * dy / dt;
            }
        }
        gradient[{ sample, static_cast<int>(column + station) }] = scale * total;
    }

} // of namespace cudaTimeShift_kernels


template <typename real_type>
void altar::models::seismic::cudaTimeShift::
shift(matrix_view_t<real_type> in, matrix_view_t<real_type> theta, matrix_view_t<real_type, false> out,
    const size_t column, const size_t traces, const size_t length, const real_type dt, cudaStream_t stream)
{
    auto cells = in.packing().shape()[0] * in.packing().shape()[1];
    cudaTimeShift_kernels::_shift<real_type><<<IDIVUP(cells, NTHREADS), NTHREADS, 0, stream>>>(
        in, theta, out, column, traces, length, dt);
    cudaCheckError("cudaTimeShift::shift error");
}

template <typename real_type>
void altar::models::seismic::cudaTimeShift::
adjoint(matrix_view_t<real_type> in, matrix_view_t<real_type> theta, matrix_view_t<real_type, false> out,
    const size_t column, const size_t traces, const size_t length, const real_type dt, cudaStream_t stream)
{
    auto cells = in.packing().shape()[0] * in.packing().shape()[1];
    cudaTimeShift_kernels::_adjoint<real_type><<<IDIVUP(cells, NTHREADS), NTHREADS, 0, stream>>>(
        in, theta, out, column, traces, length, dt);
    cudaCheckError("cudaTimeShift::adjoint error");
}

template <typename real_type>
void altar::models::seismic::cudaTimeShift::
gradient(matrix_view_t<real_type> x, matrix_view_t<real_type> w, matrix_view_t<real_type> theta,
    matrix_view_t<real_type, false> gradient,
    const size_t column, const size_t traces, const size_t length, const real_type dt,
    const real_type scale, cudaStream_t stream)
{
    auto stations = x.packing().shape()[1] / (traces * length);
    auto cells = x.packing().shape()[0] * stations;
    cudaTimeShift_kernels::_gradient<real_type><<<IDIVUP(cells, NTHREADS), NTHREADS, 0, stream>>>(
        x, w, theta, gradient, column, traces, length, dt, scale, stations);
    cudaCheckError("cudaTimeShift::gradient error");
}

#define INSTANTIATE(real_type) \
    template void altar::models::seismic::cudaTimeShift::shift<real_type>( \
        matrix_view_t<real_type>, matrix_view_t<real_type>, matrix_view_t<real_type, false>, \
        const size_t, const size_t, const size_t, const real_type, cudaStream_t); \
    template void altar::models::seismic::cudaTimeShift::adjoint<real_type>( \
        matrix_view_t<real_type>, matrix_view_t<real_type>, matrix_view_t<real_type, false>, \
        const size_t, const size_t, const size_t, const real_type, cudaStream_t); \
    template void altar::models::seismic::cudaTimeShift::gradient<real_type>( \
        matrix_view_t<real_type>, matrix_view_t<real_type>, matrix_view_t<real_type>, \
        matrix_view_t<real_type, false>, const size_t, const size_t, const size_t, const real_type, \
        const real_type, cudaStream_t);

INSTANTIATE(float)
INSTANTIATE(double)

// end of file
