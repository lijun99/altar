// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-2025 parasim inc
// (c) 2010-2025 california institute of technology
// all rights reserved

// code guard
#pragma once

// externals
#include "external.h"

// the norm bindings
namespace altar::cuda::extensions::norms {
    // build the {norms} submodule
    auto __init__(py::module &) -> void;
} // namespace altar::cuda::extensions::norms

// end of file
