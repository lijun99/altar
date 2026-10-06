// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//

// externals
#include "external.h"
// my declarations
#include "grids.h"


namespace {
    using altar::cuda::extensions::grid_t;
    namespace py = pybind11;

    // the bytes of {g}'s cells, which must be contiguous, in row major order
    auto bytes(grid_t & g, const char * routine) -> std::size_t
    {
        const auto info = g.view();
        auto stride = info.itemsize;
        for (auto i = info.ndim; i-- > 0;) {
            if (info.strides[i] != stride) {
                throw py::value_error(std::string(routine) + ": the grid is not contiguous");
            }
            stride *= info.shape[i];
        }
        return static_cast<std::size_t>(info.size * info.itemsize);
    }
}


auto
altar::cuda::extensions::grids::__init__(py::module & m) -> void
{
    auto grids = m.def_submodule("grids", "whole-grid copies and fills on the device");

    // target = source, cell for cell, on the device
    grids.def(
        "copy",
        [](grid_t & target, grid_t & source) -> void {
            if (target.view().format != source.view().format) {
                throw py::value_error("grids.copy: the grids hold different cell types");
            }
            const auto size = bytes(target, "grids.copy");
            if (bytes(source, "grids.copy") != size) {
                throw py::value_error("grids.copy: the grids hold different numbers of cells");
            }
            cudaSafeCall(cudaMemcpy(
                reinterpret_cast<void *>(target.address()),
                reinterpret_cast<const void *>(source.address()), size, cudaMemcpyDefault));
            synchronize("grids.copy");
        },
        "target"_a, "source"_a, "overwrite the cells of {target} with those of {source}");

    // target = 0, on the device
    grids.def(
        "zero",
        [](grid_t & target) -> void {
            cudaSafeCall(cudaMemset(
                reinterpret_cast<void *>(target.address()), 0, bytes(target, "grids.zero")));
            synchronize("grids.zero");
        },
        "target"_a, "fill the cells of {target} with zeroes");
}

// end of file
