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

// build the {distributions} submodule and populate it with every ported distribution's own
// bindings
namespace altar::cuda::extensions::distributions {
    auto __init__(py::module & m) -> void;
}

// end of file
