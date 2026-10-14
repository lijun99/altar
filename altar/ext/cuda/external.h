// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//

// code guard
#pragma once

// STL
#include <cstdlib>
#include <string>
#include <stdexcept>

// pybind11
#include <pybind11/pybind11.h>

// the type-erased grid these bindings operate on; already registered as a python type by
// pyre's own {pyre.grid} extension module, so binding a function against it here doesn't
// redeclare the type, just reuses it
#include <pyre/py/grid/AnyGrid.h>

// pyre's own statically typed grid, the thing {grid_t} (a {pyre::py::grid::AnyGrid}) itself
// type-erases; {pyre::memory::View} is device-annotated and non-owning
#include <pyre/grid.h>
#include <pyre/memory.h>

// shared NTHREADS/IDIVUP/cudaCheckError/cudaSafeCall
#include <altar/cuda/support.h>


// type aliases
namespace altar::cuda::extensions {
    // import {pybind11}
    namespace py = pybind11;
    // get the special {pybind11} literals
    using namespace py::literals;

    // the type-erased grid the bindings hand cell data through
    using grid_t = pyre::py::grid::AnyGrid;

    // launches stay asynchronous, since pyre's managed grids wait for the device before the host
    // reads them; set ALTAR_CUDA_SYNC to wait after every binding, to locate a failing kernel
    inline auto
    synchronize(const char * routine) -> void
    {
        static const bool eager = std::getenv("ALTAR_CUDA_SYNC") != nullptr;
        auto status = eager ? cudaDeviceSynchronize() : cudaPeekAtLastError();
        if (status != cudaSuccess) {
            throw std::runtime_error(std::string(routine) + ": " + cudaGetErrorString(status));
        }
    }

    // a device-usable, non-owning {pyre::grid::Grid<Canonical<Rank>, View<T,isConst>>} over a
    // {grid_t}'s cells, from its {address}/{shape}/{strides}; {regrid<const double,2>} is
    // read-only; {T} is not checked against the grid's format, the binding picks it; the
    // {grid_t} must outlive the result
    template <class T, std::size_t Rank>
    auto
    regrid(grid_t & g) -> pyre::grid::grid_t<
        pyre::grid::canonical_t<Rank>,
        pyre::memory::View<std::remove_const_t<T>, std::is_const_v<T>>>
    {
        using packing_t = pyre::grid::canonical_t<Rank>;
        using cell_t = std::remove_const_t<T>;
        using storage_t = pyre::memory::View<cell_t, std::is_const_v<T>>;
        using grid_type = pyre::grid::grid_t<packing_t, storage_t>;
        using shape_t = typename packing_t::shape_type;
        using strides_t = typename packing_t::strides_type;
        using difference_t = typename packing_t::difference_type;

        const auto & shape = g.shape();
        const auto & strides = g.strides();
        if (shape.size() != Rank || strides.size() != Rank) {
            throw py::value_error(
                "regrid: a rank " + std::to_string(Rank) + " view was requested of a rank "
                + std::to_string(shape.size()) + " grid");
        }

        auto sh = shape_t{};
        auto st = strides_t{};
        // {std::size_t}, to match {Rank}
        for (std::size_t i = 0; i < Rank; ++i) {
            sh[i] = shape[i];
            st[i] = strides[i];
        }
        auto packing = packing_t(sh, packing_t::index_type::zero(), packing_t::order_type::c(), st, 0);

        // a conservative capacity bound for {View}: the farthest reachable offset plus one;
        // correct whether the grid is tightly packed or a strided/sliced sub-grid
        auto capacity = difference_t{ 1 };
        for (std::size_t i = 0; i < Rank; ++i) {
            capacity += (shape[i] - 1) * strides[i];
        }

        auto view = storage_t(reinterpret_cast<T *>(g.address()), capacity, 1);
        return grid_type(packing, view);
    }

} // namespace altar::cuda::extensions

// end of file
