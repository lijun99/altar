// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//

// my declarations
#include "timeshift.h"

// the kernel launchers
#include <altar/models/seismic/cuda/cudaTimeShift.h>


auto
altar::models::seismic::extensions::timeshift::__init__(py::module & m) -> void
{
    namespace kernels = altar::models::seismic::cudaTimeShift;

    // the time shift of the traces of each station, and its adjoint
    for (auto adjoint : { false, true }) {
        m.def(
            adjoint ? "cudaTimeShift_adjoint" : "cudaTimeShift_shift",
            [adjoint](grid_t & in, grid_t & theta, grid_t & out, std::size_t column, std::size_t traces,
                      std::size_t length, double dt) -> void {
                auto format = in.view().format;
                if (format.size() == 1 && format[0] == 'd') {
                    auto launch = adjoint ? kernels::adjoint<double> : kernels::shift<double>;
                    launch(regrid<const double, 2>(in), regrid<const double, 2>(theta), regrid<double, 2>(out),
                           column, traces, length, dt, 0);
                } else if (format.size() == 1 && format[0] == 'f') {
                    auto launch = adjoint ? kernels::adjoint<float> : kernels::shift<float>;
                    launch(regrid<const float, 2>(in), regrid<const float, 2>(theta), regrid<float, 2>(out),
                           column, traces, length, static_cast<float>(dt), 0);
                } else {
                    throw py::value_error("cudaTimeShift: unsupported grid cell type '" + format + "'");
                }
                cudaCheckError("cudaTimeShift");
                synchronize("cudaTimeShift");
            },
            "in"_a, "theta"_a, "out"_a, "column"_a, "traces"_a, "length"_a, "dt"_a,
            adjoint ? "out = the adjoint of the time shift of the stations, applied to in"
                    : "out = in, the traces of each station shifted by theta[:, column + station]");
    }

    // the derivatives of w . shift(x) with respect to the shifts, times {scale}
    m.def(
        "cudaTimeShift_gradient",
        [](grid_t & x, grid_t & w, grid_t & theta, grid_t & gradient, std::size_t column,
           std::size_t traces, std::size_t length, double dt, double scale) -> void {
            auto format = x.view().format;
            if (format.size() == 1 && format[0] == 'd') {
                kernels::gradient<double>(regrid<const double, 2>(x), regrid<const double, 2>(w),
                    regrid<const double, 2>(theta), regrid<double, 2>(gradient),
                    column, traces, length, dt, scale);
            } else if (format.size() == 1 && format[0] == 'f') {
                kernels::gradient<float>(regrid<const float, 2>(x), regrid<const float, 2>(w),
                    regrid<const float, 2>(theta), regrid<float, 2>(gradient),
                    column, traces, length, static_cast<float>(dt), static_cast<float>(scale));
            } else {
                throw py::value_error("cudaTimeShift_gradient: unsupported grid cell type '" + format + "'");
            }
            cudaCheckError("cudaTimeShift_gradient");
            synchronize("cudaTimeShift_gradient");
        },
        "x"_a, "w"_a, "theta"_a, "gradient"_a, "column"_a, "traces"_a, "length"_a, "dt"_a, "scale"_a,
        "gradient[:, column + k] = scale * the derivative of w . shift(x) with respect to the shift of station k");
}

// end of file
