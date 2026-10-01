// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2022 - 2023 california institute of technology
// all rights reserved

// code guard
#ifndef altar_models_seas_ext_cudaseas_external_h
#define altar_models_seas_ext_cudaseas_external_h


// pybind11
#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

// cuda error checking
#include <pyre/cuda.h>

// the type-erased grid the bindings receive cell data through
#include <pyre/py/grid/AnyGrid.h>

// type aliases
namespace altar::cuda::py::seas {
    // import {pybind11}
    namespace py = pybind11;
    // get the special {pybind11} literals
    using namespace py::literals;

    // the type-erased grid
    using grid_t = pyre::py::grid::AnyGrid;

    // the cells of {g} as a raw device pointer, checked against the size of {T}
    template <typename T>
    inline auto cells(grid_t & g) -> T *
    {
        if (g.view().itemsize != static_cast<py::ssize_t>(sizeof(T))) {
            throw py::type_error(
                "seas: expected a grid with " + std::to_string(sizeof(T)) + "-byte cells, got "
                + std::to_string(g.view().itemsize) + "-byte ones");
        }
        return reinterpret_cast<T *>(g.address());
    }

    namespace linearviscous {
        namespace py = pybind11;
    }

    namespace ratedependent {
        namespace py = pybind11;
    }

    namespace tractiondependent {
        namespace py = pybind11;
    }

} // namespace

#endif
// end of file
