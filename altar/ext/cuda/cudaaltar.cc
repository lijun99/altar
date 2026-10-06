// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//

// externals
#include "external.h"

// the module method declarations
#include "metadata.h"
#include "norm.h"
#include "distributions.h"
#include "metropolis.h"
#include "leapfrog.h"
#include "langevin.h"
#include "grids.h"

// the module entry point
PYBIND11_MODULE(cudaaltar, m)
{
    // the docstring
    m.doc() = "altar cuda extension module";

    // module metadata
    altar::cuda::extensions::metadata(m);
    // norms
    altar::cuda::extensions::norms::__init__(m);
    // distributions
    altar::cuda::extensions::distributions::__init__(m);
    // the metropolis-hastings accept/reject step
    altar::cuda::extensions::metropolis::__init__(m);
    // the leapfrog integrator for hamiltonian monte carlo
    altar::cuda::extensions::leapfrog::__init__(m);
    // stochastic gradient langevin dynamics (SGLD)
    altar::cuda::extensions::langevin::__init__(m);
    // whole-grid copies and fills on the device
    altar::cuda::extensions::grids::__init__(m);
}


// end of file
