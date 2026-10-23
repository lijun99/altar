// -*- C++ -*-
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//

#pragma once

#include <cuda_runtime.h>
#include <altar/cuda/support.h>

// the time shift of each station's traces, by Keys cubic interpolation: the observations are the
// traces of the stations, {traces} per station and {length} samples each, station by station;
// the shift of station k, in seconds, is theta[s, column + k], {dt} the sampling interval; a
// positive shift delays the traces, y[i] = x(i - shift/dt), with zeros beyond their ends
namespace altar::models::seismic::cudaTimeShift {

    // out = the shifted {in}; an assignment
    template <typename real_type>
    void shift(matrix_view_t<real_type> in, matrix_view_t<real_type> theta,
        matrix_view_t<real_type, false> out,
        const size_t column, const size_t traces, const size_t length, const real_type dt,
        cudaStream_t stream=0);

    // out = the adjoint shift of {in}, the transpose of {shift}; an assignment
    template <typename real_type>
    void adjoint(matrix_view_t<real_type> in, matrix_view_t<real_type> theta,
        matrix_view_t<real_type, false> out,
        const size_t column, const size_t traces, const size_t length, const real_type dt,
        cudaStream_t stream=0);

    // gradient[s, column + k] = scale * sum over the traces of station k of w . d(shift x)/d(shift_k);
    // an assignment
    template <typename real_type>
    void gradient(matrix_view_t<real_type> x, matrix_view_t<real_type> w, matrix_view_t<real_type> theta,
        matrix_view_t<real_type, false> gradient,
        const size_t column, const size_t traces, const size_t length, const real_type dt,
        const real_type scale, cudaStream_t stream=0);

} // of namespace altar::models::seismic::cudaTimeShift


// end of file
