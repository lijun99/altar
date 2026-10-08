// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//

#include "bindings.h"


PYBIND11_MODULE(mogi, m)
{
    m.doc() = "the altar mogi extension module";
    altar::models::mogi::extension::bindings(m);
}

// end of file
