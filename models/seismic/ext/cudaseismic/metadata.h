// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Lijun Zhu
// california institute of technology
// (c) 2016-2020  all rights reserved
//

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
