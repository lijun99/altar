// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//

// pybind11, grids, and the shared binding helpers
#include "external.h"
// the kernel launchers
#include <altar/models/mogi/cuda/cudaMogi.h>

using namespace altar::cuda::extensions;


PYBIND11_MODULE(cudamogi, m)
{
    m.doc() = "the altar mogi cuda extension module";

    m.def(
        "displacements",
        [](grid_t & theta, grid_t & stations,
           std::size_t xIdx, std::size_t yIdx, std::size_t dIdx, std::size_t sIdx,
           bool log10dV, double nu, std::size_t batch, grid_t & predicted) -> void {
            namespace kernels = altar::models::mogi::cuda;
            auto format = theta.view().format;
            if (format.size() == 1 && format[0] == 'd') {
                kernels::displacements<double>(
                    regrid<const double, 2>(theta), regrid<const double, 2>(stations),
                    xIdx, yIdx, dIdx, sIdx, log10dV, nu, batch, regrid<double, 2>(predicted));
            } else if (format.size() == 1 && format[0] == 'f') {
                kernels::displacements<float>(
                    regrid<const float, 2>(theta), regrid<const float, 2>(stations),
                    xIdx, yIdx, dIdx, sIdx, log10dV, static_cast<float>(nu), batch,
                    regrid<float, 2>(predicted));
            } else {
                throw py::value_error("cudamogi.displacements: unsupported grid cell type '" + format + "'");
            }
            synchronize("cudamogi.displacements");
        },
        "theta"_a, "stations"_a, "xIdx"_a, "yIdx"_a, "dIdx"_a, "sIdx"_a,
        "log10dV"_a, "nu"_a, "batch"_a, "predicted"_a,
        "fill predicted[:batch] with the LOS displacements of the Mogi sources in theta[:batch]");
}

// end of file
