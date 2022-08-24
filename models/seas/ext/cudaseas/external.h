// -*- c++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// (c) 1998-2022 all rights reserved

// code guard
#if !defined(altar_models_seas_ext__cudaseas_external_h)
#define altar_models_seas_ext__cudaseas_external_h


// pybind11
#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

// cuda modules for vector/matrix
#include <pyre/cuda.h>
#include <pyre/cuda/capsules.h>

// type aliases
namespace altar::cuda::py::seas {
    // import {pybind11}
    namespace py = pybind11;
    // get the special {pybind11} literals
    using namespace py::literals;

    namespace linearviscous {
        namespace py = pybind11;
    }

} // namespace

#endif
// end of file

