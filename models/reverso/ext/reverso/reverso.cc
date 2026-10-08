// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//

#include "bindings.h"


PYBIND11_MODULE(reverso, m)
{
    m.doc() = "the altar reverso extension module";
    altar::models::reverso::extension::bindings(m);
}

// end of file
