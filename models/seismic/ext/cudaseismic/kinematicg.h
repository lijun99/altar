// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-2019 parasim inc
// (c) 2010-2019 california institute of technology
// all rights reserved
//
// Author(s): Lijun Zhu

#if !defined(altar_extensions_models_cudaseismic_kinematicg_h)
#define altar_extensions_models_cudaseismic_kinematicg_h

#include <altar/models/seismic/cuda/cudaKinematicG.h>

// place everything in my private namespace
namespace altar {
    namespace extensions {
        namespace models {
            namespace cudaseismic {

                  // allocate
                  extern const char * const kinematicg_alloc__name__;
                  extern const char * const kinematicg_alloc__doc__;
                  PyObject * kinematicg_alloc(PyObject *, PyObject *);

                  // free
                  void kinematicg_free(PyObject *);

                  // forward model, for one set of parameters
                  extern const char * const kinematicg_forward__name__;
                  extern const char * const kinematicg_forward__doc__;
                  PyObject * kinematicg_forward(PyObject *, PyObject *);

                  // castMb - calculate Mb(x, t) only, for one set of parameters
                  extern const char * const kinematicg_castMb__name__;
                  extern const char * const kinematicg_castMb__doc__;
                  PyObject * kinematicg_castMb(PyObject *, PyObject *);

                  // linearGM - calculate Gb x Mb only, for one set of parameters
                  extern const char * const kinematicg_linearGM__name__;
                  extern const char * const kinematicg_linearGM__doc__;
                  PyObject * kinematicg_linearGM(PyObject *, PyObject *);

                  // forward model batched
                  extern const char * const kinematicg_forward_batched__name__;
                  extern const char * const kinematicg_forward_batched__doc__;
                  PyObject * kinematicg_forward_batched(PyObject *, PyObject *);

                  using SModel_t = altar::models::seismic::cudaKinematicG<float>;
                  using DModel_t = altar::models::seismic::cudaKinematicG<double>;

            } // of namespace cudaseismic
        } // of namespace models
    } // of namespace extensions
} // of namespace altar

#endif

// end of file
