// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-2025 parasim inc
// (c) 2010-2025 california institute of technology
// all rights reserved

// code guard
#pragma once

#include "external.h"

// {cudaRanged}'s own bindings (support/range verification and clamping, shared by uniform-
// like distributions), added to the (already created) {distributions} submodule
namespace altar::cuda::extensions::distributions::ranged {
    auto __init__(py::module & distributions) -> void;
}

// end of file
