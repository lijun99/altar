// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//

#pragma once

#include "external.h"

namespace altar::models::seismic::extensions::kinematic {
    using namespace altar::cuda::extensions;
    auto __init__(py::module & m) -> void;
}

// end of file
