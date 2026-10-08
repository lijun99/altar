// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//

// code guard
#pragma once

#include <pybind11/pybind11.h>

namespace altar::models::cdm::extension {
    // the forward model and its checks
    void bindings(pybind11::module & m);
}

// end of file
