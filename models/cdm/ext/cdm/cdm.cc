// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//

#include "bindings.h"


PYBIND11_MODULE(cdm, m)
{
    m.doc() = "the altar cdm extension module";
    altar::models::cdm::extension::bindings(m);
}

// end of file
