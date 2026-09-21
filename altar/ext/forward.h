// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// (c) 2013-2026 all rights reserved

// code guard
#pragma once


// external dependencies, and the aliases that shape this namespace
#include "external.h"


// the {altar} extension namespace
namespace altar::py {
    // what the package says about itself
    void metadata(py::module & m);
    // matrix conditioning
    void condition(py::module & m);
    // the annealing schedule: {COV}, {dbeta_brent}, {dbeta_grid}, {low_variance_random}
    void dbeta(py::module & m);
} // namespace altar::py


// end of file
