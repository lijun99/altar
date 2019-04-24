// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-2019 parasim inc
// (c) 2010-2019 california institute of technology
// all rights reserved
//
// Author(s): Lijun Zhu


#if !defined(cualtar_extensions_distributions_h)
#define cualtar_extensions_distributions_h

#include "capsules.h"

// place everything in my private namespace
namespace altar { namespace cuda { namespace extensions { 
        // uniform distribution        
        namespace cudaUniform {

            extern const char * const sample__name__;
            extern const char * const sample__doc__;
            PyObject * sample(PyObject *, PyObject *);

            extern const char * const logpdf__name__;
            extern const char * const logpdf__doc__;
            PyObject * logpdf(PyObject *, PyObject *);

            extern const char * const verify__name__;
            extern const char * const verify__doc__;
            PyObject * verify(PyObject *, PyObject *);
        } 
        
        // gaussian distribution
        namespace cudaGaussian {
            
            extern const char * const sample__name__;
            extern const char * const sample__doc__;
            PyObject * sample(PyObject *, PyObject *);

            extern const char * const logpdf__name__;
            extern const char * const logpdf__doc__;
            PyObject * logpdf(PyObject *, PyObject *);
        }

} } } // of namespace altar.cuda.extensions

#endif

// end of file
