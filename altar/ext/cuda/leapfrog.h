// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-2025 parasim inc
// (c) 2010-2025 california institute of technology
// all rights reserved

// code guard
#pragma once

#include "external.h"

// the leapfrog integrator's own bindings (hamiltonian monte carlo)
namespace altar::cuda::extensions::leapfrog {
    auto __init__(py::module & m) -> void;
}

// end of file
