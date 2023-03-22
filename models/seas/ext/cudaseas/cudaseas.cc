// -*- c++ -*-
//
// (c) 1998-2022 all rights reserved

// externals
#include "external.h"

// namespace setup
#include "forward.h"



// the module entry point
PYBIND11_MODULE(cudaseas, m)
{
    // the doc string
    m.doc() = "the cuda extension module for seas model methods";

    // the linear viscous method bindings
    auto ms = m.def_submodule("linearviscous", "linear viscous ode binding");
    altar::cuda::py::seas::linearviscous::module(ms);

}

// end of file
