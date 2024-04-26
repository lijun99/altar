// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-2021 parasim inc
// (c) 2010-2021 california institute of technology
// all rights reserved
//
// Author(s): Lijun Zhu

#if !defined(altar_extensions_models_cudaseismic_static_h)
#define altar_extensions_models_cudaseismic_static_h

// place everything in my private namespace
namespace altar {
    namespace extensions {
        namespace models {
            namespace cudaseismic {

                  // allocate
                  extern const char * const static_gemm_col__name__;
                  extern const char * const static_gemm_col__doc__;
                  PyObject * static_gemm_col(PyObject *, PyObject *);

            } // of namespace cudaseismic
        } // of namespace models
    } // of namespace extensions
} // of namespace altar

#endif

// end of file
