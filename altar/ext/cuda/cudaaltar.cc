// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-2025 parasim inc
// (c) 2010-2025 california institute of technology
// all rights reserved
//
// Author(s): Lijun Zhu

// externals
#include "external.h"

// the module method declarations
#include "metadata.h"
#include "norm.h"
#include "distributions.h"
#include "metropolis.h"
#include "leapfrog.h"
#include "langevin.h"

// within {distributions}: {tgaussianlogit}/{uniformlogit} are not ported -- their old
// bindings' {sample}/{logpdf}/{logpdfgradient} were declared but never actually defined
// anywhere (a link error waiting to happen, not a working path to port), matching the
// reparameterization work the user explicitly deferred earlier ("let's work on the logistic
// or reparameterization later")
//
// within {distributions}: {gaussian}, {uniform}, {ranged}, {tgaussian}, {logistic} are
// ported; {tgaussianlogit}/{uniformlogit} are not -- their old bindings' {sample}/{logpdf}/
// {logpdfgradient} were declared but never actually defined anywhere (a link error waiting
// to happen, not a working path to port), matching the reparameterization work the user
// explicitly deferred earlier ("let's work on the logistic or reparameterization later")


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
}


// end of file
