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
#include "distributions.h"
#include "norm.h"
#include "metropolis.h"

// put everything in my private namespace
namespace altar { namespace cuda {
    namespace extensions {
        // the module method table
        PyMethodDef module_methods[] = {
            // module metadata
            // the copyright method
            { copyright__name__, copyright, METH_VARARGS, copyright__doc__ },
            // the license
            { license__name__, license, METH_VARARGS, license__doc__ },
            // the version
            { version__name__, version, METH_VARARGS, version__doc__ },

            // distributions
            { cudaRanged::verify__name__, cudaRanged::verify, METH_VARARGS, cudaRanged::verify__doc__},
            // cudaUniform
            { cudaUniform::sample__name__, cudaUniform::sample, METH_VARARGS, cudaUniform::sample__doc__},
            { cudaUniform::logpdf__name__, cudaUniform::logpdf, METH_VARARGS, cudaUniform::logpdf__doc__},
            // cudaGaussian
            { cudaGaussian::sample__name__, cudaGaussian::sample, METH_VARARGS, cudaGaussian::sample__doc__},
            { cudaGaussian::logpdf__name__, cudaGaussian::logpdf, METH_VARARGS, cudaGaussian::logpdf__doc__},
            // cudaTGaussian
            { cudaTGaussian::sample__name__, cudaTGaussian::sample, METH_VARARGS, cudaTGaussian::sample__doc__},
            { cudaTGaussian::logpdf__name__, cudaTGaussian::logpdf, METH_VARARGS, cudaTGaussian::logpdf__doc__},

            // norms
            { cudaL2::norm__name__, cudaL2::norm, METH_VARARGS, cudaL2::norm__doc__},
            { cudaL2::normllk__name__, cudaL2::normllk, METH_VARARGS, cudaL2::normllk__doc__},

            // metropolis sampler
            { cudaMetropolis::setValidSampleIndices__name__, cudaMetropolis::setValidSampleIndices, METH_VARARGS, cudaMetropolis::setValidSampleIndices__doc__},
            { cudaMetropolis::queueValidSamples__name__, cudaMetropolis::queueValidSamples, METH_VARARGS, cudaMetropolis::queueValidSamples__doc__},
            { cudaMetropolis::metropolisUpdate__name__, cudaMetropolis::metropolisUpdate, METH_VARARGS, cudaMetropolis::metropolisUpdate__doc__},


            // sentinel
            {0, 0, 0, 0}
        };

        // the module documentation string
        const char * const __doc__ = "altar cuda extension module";

        // the module definition structure
        PyModuleDef module_definition = {
            // header
            PyModuleDef_HEAD_INIT,
            // the name of the module
            "cudaaltar",
            // the module documentation string
            __doc__,
            // size of the per-interpreter state of the module; -1 if this state is global
            -1,
            // the methods defined in this module
            module_methods
        };
    } // of namespace extensions
} } // of namespace altar.cuda


// initialization function for the module
// *must* be called PyInit_cudaaltar
PyMODINIT_FUNC
PyInit_cudaaltar()
{
    // create the module
    PyObject * module = PyModule_Create(&altar::cuda::extensions::module_definition);
    // check whether module creation succeeded and raise an exception if not
    if (!module) {
        return 0;
    }
    // otherwise, we have an initialized module
    // return the newly created module
    return module;
}

// end of file
