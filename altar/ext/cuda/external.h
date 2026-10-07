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

// device-annotated mdspan, bundled with the cuda toolkit's own CCCL; unlike the standard
// library's {std::mdspan} (C++23, and unusable in device code on this nvcc, which caps at
// {--std=c++20}), this one is usable inside kernels
#include <cuda/std/mdspan>

// pyre's own statically typed grid, the thing {grid_t} (a {pyre::py::grid::AnyGrid}) itself
// type-erases; {pyre::memory::View} is device-annotated and non-owning, same as {mdspan}
#include <pyre/grid.h>
#include <pyre/memory.h>

// shared NTHREADS/IDIVUP/cudaCheckError/cudaSafeCall
#include <altar/cuda/support.h>

// an alias for CCCL's {::cuda::std}, at global scope: inside {namespace altar::cuda::...}, an
// unqualified {cuda::std::whatever} would resolve {cuda} to the enclosing {altar::cuda}
// namespace, not the global one CCCL lives in
namespace cudastd = ::cuda::std;


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

    // an ephemeral, device-usable view over a {grid_t}'s cells, in terms of {Rank} (a compile
    // time constant, since {grid_t} is rank-erased but {mdspan} isn't) and {T} (the caller's
    // choice, once it has picked a branch on the grid's runtime cell format -- {span} doesn't
    // itself check {T} against the grid's format string, matching the existing convention of
    // checking format once, in the binding, before calling into any cell-type-templated code)
    //
    // {layout_stride}, not the default {layout_right}, because a {grid_t} can be a sliced,
    // non-contiguous sub-grid; this handles both the sliced and the contiguous case uniformly,
    // at no extra cost when the grid happens to be contiguous
    //
    // non-owning, just like {grid_t.address()} itself: the caller's {grid_t} must outlive the
    // returned {mdspan}
    template <class T, std::size_t Rank>
    auto
    span(grid_t & g) -> cudastd::mdspan<T, cudastd::dextents<std::ptrdiff_t, Rank>, cudastd::layout_stride>
    {
        using extents_t = cudastd::dextents<std::ptrdiff_t, Rank>;
        using mapping_t = cudastd::layout_stride::mapping<extents_t>;
        using view_t = cudastd::mdspan<T, extents_t, cudastd::layout_stride>;
        using array_t = cudastd::array<std::ptrdiff_t, Rank>;

        const auto & shape = g.shape();
        const auto & strides = g.strides();
        if (shape.size() != Rank || strides.size() != Rank) {
            throw py::value_error(
                "span: a rank " + std::to_string(Rank) + " view was requested of a rank "
                + std::to_string(shape.size()) + " grid");
        }

        auto extentValues = array_t{};
        auto strideValues = array_t{};
        // {i} stays {std::size_t}, not {auto i = 0}: it's compared against {Rank}, itself a
        // {std::size_t} template argument, and {auto i = 0} would deduce {int}, an unsigned
        // vs. signed comparison the compiler would (rightly) flag
        for (std::size_t i = 0; i < Rank; ++i) {
            extentValues[i] = shape[i];
            strideValues[i] = strides[i];
        }

        return view_t{reinterpret_cast<T *>(g.address()), mapping_t{extents_t{extentValues}, strideValues}};
    }

    // the same conversion as {span}, but landing on pyre's own grid type instead of {mdspan}:
    // a {pyre::grid::Grid<Canonical<Rank>, View<T,isConst>>}, reconstructed from exactly the
    // same three values ({address}/{shape}/{strides}) a {grid_t} exposes. {isConst} is read off
    // {T} ({regrid<const double,2>} for a read-only view), matching how {span}'s {T} already
    // carries constness for {mdspan}.
    //
    // preferred over {span} going forward: it's pyre's own vocabulary (the same one {grid_t}
    // itself erases, and the same one already used for "grids on cuda storage in kernels" in
    // pyre's own examples), so host and device code share one grid type and one indexing style
    // ({g[{i,j}]}) instead of introducing {mdspan} as a second, CCCL-specific one. {span}/
    // {mdspan} stay available above, not removed, in case a caller wants the {mdspan} form
    // specifically (e.g. to hand a view to code that already speaks {mdspan})
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
        // {i} stays {std::size_t}, for the same reason as in {span} above
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
