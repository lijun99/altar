// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//
// hailiang zhang
// externals
#include "external.h"
// my declarations
#include "norm.h"

// the kernel launchers
#include <altar/cuda/norm/cudaL2.h>


// build the {norms} submodule
auto
altar::cuda::extensions::norms::__init__(py::module & m) -> void
{
    // make the submodule
    auto norms = m.def_submodule("norms", "the l2 norm of a batch of data");

    // ||data||, one row of {data} per sample
    norms.def(
        "cudaL2_norm",
        [](grid_t & data, grid_t & result, std::size_t batch) -> void {
            auto format = data.view().format;
            if (format.size() == 1 && format[0] == 'd') {
                altar::cuda::norms::cudaL2::norm<double>(
                    regrid<const double, 2>(data), regrid<double, 1>(result), batch);
            } else if (format.size() == 1 && format[0] == 'f') {
                altar::cuda::norms::cudaL2::norm<float>(
                    regrid<const float, 2>(data), regrid<float, 1>(result), batch);
            } else {
                throw py::value_error("cudaL2_norm: unsupported grid cell type '" + format + "'");
            }
            cudaCheckError("cudaL2_norm");
            synchronize("cudaL2_norm");
        },
        "data"_a, "result"_a, "batch"_a,
        "fill {result} with the l2 norm of the first {batch} rows of {data}");

    // constant - 0.5 ||data||^2, the l2 log likelihood
    norms.def(
        "cudaL2_normllk",
        [](grid_t & data, grid_t & result, std::size_t batch, double constant) -> void {
            auto format = data.view().format;
            if (format.size() == 1 && format[0] == 'd') {
                altar::cuda::norms::cudaL2::normllk<double>(
                    regrid<const double, 2>(data), regrid<double, 1>(result), batch, constant);
            } else if (format.size() == 1 && format[0] == 'f') {
                altar::cuda::norms::cudaL2::normllk<float>(
                    regrid<const float, 2>(data), regrid<float, 1>(result), batch,
                    static_cast<float>(constant));
            } else {
                throw py::value_error("cudaL2_normllk: unsupported grid cell type '" + format + "'");
            }
            cudaCheckError("cudaL2_normllk");
            synchronize("cudaL2_normllk");
        },
        "data"_a, "result"_a, "batch"_a, "constant"_a,
        "fill {result} with {constant} - 0.5 * ||data||^2 for the first {batch} rows of {data}");
}


// end of file
