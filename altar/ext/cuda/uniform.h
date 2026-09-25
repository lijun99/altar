// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-2025 parasim inc
// (c) 2010-2025 california institute of technology
// all rights reserved

// code guard
#pragma once

#include "external.h"

// the uniform distribution's own bindings, added to the (already created) {distributions}
// submodule
namespace altar::cuda::extensions::distributions::uniform {
    auto __init__(py::module & distributions) -> void;
}

// end of file
