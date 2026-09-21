// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// (c) 2013-2026 all rights reserved

// code guard
#pragma once


// STL
#include <string>
#include <tuple>
// support
#include <pyre/journal.h>
// the libaltar C++ library we wrap
#include <altar/bayesian/COV.h>
// the library we build on
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_rng.h>
// pybind11
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


// type aliases
namespace altar::py {
    // import {pybind11}
    namespace py = pybind11;
    // get the special {pybind11} literals, so that argument names can be spelled "name"_a
    using namespace py::literals;

    // for decorating pybind11 classes
    // class names
    using classname_t = const char *;
    // docstrings
    using docstring_t = const char *;
} // namespace altar::py


// end of file
