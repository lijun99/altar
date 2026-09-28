// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//

#include "external.h"

#include "moment.h"
#include "kinematic.h"


PYBIND11_MODULE(cudaseismic, m)
{
    // the docstring
    m.doc() = "altar seismic cuda extension module";

    // the moment magnitude prior
    altar::models::seismic::extensions::moment::__init__(m);
    // the kinematic slip model
    altar::models::seismic::extensions::kinematic::__init__(m);
}

// end of file
