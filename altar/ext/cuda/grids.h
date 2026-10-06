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

// whole-grid copies and fills on the device, so managed cells stay put
namespace altar::cuda::extensions::grids {
    auto __init__(py::module & m) -> void;
}

// end of file
