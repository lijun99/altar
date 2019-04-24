// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Lijun Zhu
// california institute of technology
// (c) 2016-2019  all rights reserved
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
