// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//

#include <portinfo>
#include <Python.h>

#include "metadata.h"
#include <altar/models/seismic/cuda/version.h>

// version
const char * const
altar::extensions::models::cudaseismic::version__name__ = "version";

const char * const
altar::extensions::models::cudaseismic::version__doc__ = "the module version string";

PyObject *
altar::extensions::models::cudaseismic::
version(PyObject *, PyObject *)
{
    return Py_BuildValue("s", altar::models::seismic::cuda::version());
}


// end of file
