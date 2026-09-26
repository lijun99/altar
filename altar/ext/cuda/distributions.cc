// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//

// externals
#include "external.h"
// my declarations
#include "distributions.h"

// each distribution's own bindings
#include "gaussian.h"
#include "uniform.h"
#include "ranged.h"
#include "tgaussian.h"
#include "logistic.h"
#include "logittransform.h"


// build the {distributions} submodule once, here, and hand it to each distribution's own
// {__init__} to populate -- pybind11's {def_submodule} makes a fresh module object every time
// it's called, so calling it once per distribution file would silently make each one clobber
// the last rather than share a single "distributions" namespace
auto
altar::cuda::extensions::distributions::__init__(py::module & m) -> void
{
    auto distributions = m.def_submodule("distributions", "cuda distribution kernels");

    altar::cuda::extensions::distributions::gaussian::__init__(distributions);
    altar::cuda::extensions::distributions::uniform::__init__(distributions);
    altar::cuda::extensions::distributions::ranged::__init__(distributions);
    altar::cuda::extensions::distributions::tgaussian::__init__(distributions);
    altar::cuda::extensions::distributions::logistic::__init__(distributions);
    altar::cuda::extensions::distributions::logittransform::__init__(distributions);
}


// end of file
