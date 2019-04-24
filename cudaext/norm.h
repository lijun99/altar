// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-2019 parasim inc
// (c) 2010-2019 california institute of technology
// all rights reserved
//
// Author(s): Lijun Zhu


#if !defined(cualtar_extensions_norm_h)
#define cualtar_extensions_norm_h


// place everything in my private namespace
namespace altar{ namespace cuda { namespace extensions { 
        // norm      
        namespace cudaL2 {
            extern const char * const norm__name__;
            extern const char * const norm__doc__;
            PyObject * norm(PyObject *, PyObject *);
            
            extern const char * const normllk__name__;
            extern const char * const normllk__doc__;
            PyObject * normllk(PyObject *, PyObject *);
        } 
} } } // of namespace altar.cuda.extensions

#endif

// end of file
