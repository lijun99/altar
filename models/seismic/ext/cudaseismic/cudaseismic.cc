// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-2020 parasim inc
// (c) 2010-2020 california institute of technology
// all rights reserved
//
// Author(s): Lijun Zhu

// for the build system
#include <portinfo>
// external dependencies
#include <string>
#include <Python.h>

// the module method declarations
#include "metadata.h"
#include "kinematicg.h"

// put everything in my private namespace
namespace altar {
    namespace extensions {
        namespace models {
            namespace cudaseismic {
                // the module method table
                extern PyMethodDef module_methods[];
                extern PyModuleDef module_definition;
            } // of namespace cudacdm
        } // of namespace models
    } // of namespace extensions
} // of namespace altar

PyMethodDef
altar::extensions::models::cudaseismic::
module_methods[] = {
    // module metadata
    // the version
    { version__name__, version, METH_VARARGS, version__doc__ },

    // kinematicg model
    { kinematicg_alloc__name__, kinematicg_alloc, METH_VARARGS, kinematicg_alloc__doc__ },
    { kinematicg_forward__name__, kinematicg_forward, METH_VARARGS, kinematicg_forward__doc__ },
    { kinematicg_forward_batched__name__, kinematicg_forward_batched, METH_VARARGS, kinematicg_forward_batched__doc__ },
    { kinematicg_castMb__name__, kinematicg_castMb, METH_VARARGS, kinematicg_castMb__doc__ },
    { kinematicg_linearGM__name__, kinematicg_linearGM, METH_VARARGS, kinematicg_linearGM__doc__},

    // sentinel
    {0, 0, 0, 0}
};

// the module definition structure
PyModuleDef
altar::extensions::models::cudaseismic::
module_definition = {
    // header
    PyModuleDef_HEAD_INIT,
    // the name of the module
    "cudaseismic",
    // the module documentation string
    "the seismic extension module with support for CUDA",
    // size of the per-interpreter state of the module; -1 if this state is global
    -1,
    // the methods defined in this module
    module_methods
};

// initialization function for the module
// *must* be called PyInit_altar
PyMODINIT_FUNC
PyInit_cudaseismic()
{
    // create the module
    PyObject * module = PyModule_Create(&altar::extensions::models::cudaseismic::module_definition);
    // check whether module creation succeeded
    if (!module) {
        // and raise an exception if not
        return 0;
    }
    // otherwise, we have an initialized module
    // return the newly created module
    return module;
}

// end of file
