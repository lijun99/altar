// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-2025 parasim inc
// (c) 2010-2025 california institute of technology
// all rights reserved
//
// Author(s): Codex

#if !defined(cualtar_extensions_leapfrog_h)
#define cualtar_extensions_leapfrog_h

namespace altar { namespace cuda { namespace extensions {
    namespace cudaLeapfrog {

        extern const char * const sampleMomentum__name__;
        extern const char * const sampleMomentum__doc__;
        PyObject * sampleMomentum(PyObject *, PyObject *);

        extern const char * const computePotentialAndGradient__name__;
        extern const char * const computePotentialAndGradient__doc__;
        PyObject * computePotentialAndGradient(PyObject *, PyObject *);

        extern const char * const kineticEnergy__name__;
        extern const char * const kineticEnergy__doc__;
        PyObject * kineticEnergy(PyObject *, PyObject *);

        extern const char * const updatePosition__name__;
        extern const char * const updatePosition__doc__;
        PyObject * updatePosition(PyObject *, PyObject *);

        extern const char * const updateMomentum__name__;
        extern const char * const updateMomentum__doc__;
        PyObject * updateMomentum(PyObject *, PyObject *);

        extern const char * const metropolis__name__;
        extern const char * const metropolis__doc__;
        PyObject * metropolis(PyObject *, PyObject *);

        extern const char * const restoreRejected__name__;
        extern const char * const restoreRejected__doc__;
        PyObject * restoreRejected(PyObject *, PyObject *);

        extern const char * const restoreMatrix__name__;
        extern const char * const restoreMatrix__doc__;
        PyObject * restoreMatrix(PyObject *, PyObject *);
    }
}}}

#endif

// end of file
