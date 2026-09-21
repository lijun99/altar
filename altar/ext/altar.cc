// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// (c) 2013-2026 all rights reserved

// external dependencies
#include "external.h"
// namespace setup
#include "forward.h"


// the module entry point
PYBIND11_MODULE(altar, m)
{
    // the docstring
    m.doc() = "the altar extension module";

    // what the package says about itself
    altar::py::metadata(m);
    // matrix conditioning
    altar::py::condition(m);
    // the annealing schedule
    altar::py::dbeta(m);
}

// end of file
