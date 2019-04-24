// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-2019 parasim inc
// (c) 2010-2019 california institute of technology
// all rights reserved
//
// Author(s): Lijun Zhu

#if !defined(cualtar_extensions_metadata_h)
#define cualtar_extensions_metadata_h


// place everything in my private namespace
namespace altar { namespace cuda {
    namespace extensions {
        // copyright note
        extern const char * const copyright__name__;
        extern const char * const copyright__doc__;
        PyObject * copyright(PyObject *, PyObject *);

        // license
        extern const char * const license__name__;
        extern const char * const license__doc__;
        PyObject * license(PyObject *, PyObject *);

        // version
        extern const char * const version__name__;
        extern const char * const version__doc__;
        PyObject * version(PyObject *, PyObject *);

    } // of namespace extensions
} }// of namespace cualtar

#endif

// end of file
