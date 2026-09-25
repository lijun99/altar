// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//
// california institute of technology

#if !defined(altar_extensions_models_cudacdm_metadata_h)
#define altar_extensions_models_cudacdm_metadata_h


// place everything in my private namespace
namespace altar {
    namespace extensions {
        namespace models {
            namespace cudaseismic {
                // version
                extern const char * const version__name__;
                extern const char * const version__doc__;
                PyObject * version(PyObject *, PyObject *);
            } // of namespace cudaseismic
        } // of namespace models
    } // of namespace extensions
} // of namespace altar

#endif

// end of file
