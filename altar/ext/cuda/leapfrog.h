// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//

// code guard
#pragma once

#include "external.h"

// the leapfrog integrator's own bindings (hamiltonian monte carlo)
namespace altar::cuda::extensions::leapfrog {
    auto __init__(py::module & m) -> void;
}

// end of file
