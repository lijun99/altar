// -*- C++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
//
// (c) 2013-2019 parasim inc
// (c) 2010-2019 california institute of technology
// all rights reserved
//

#if !defined(altar_extensions_dbeta_h)
#define altar_extensions_dbeta_h


// place everything in my private namespace
namespace altar {
    namespace extensions {

        // cov
        extern const char * const cov__name__;
        extern const char * const cov__doc__;
        PyObject * cov(PyObject *, PyObject *);

        // dbeta
        extern const char * const dbeta_gsl__name__;
        extern const char * const dbeta_gsl__doc__;
        PyObject * dbeta_gsl(PyObject *, PyObject *);

        extern const char * const dbeta_grid__name__;
        extern const char * const dbeta_grid__doc__;
        PyObject * dbeta_grid(PyObject *, PyObject *);


        // dbeta_gsl
        extern const char * const dbeta_gsl__name__;
        extern const char * const dbeta_gsl__doc__;
        PyObject * dbeta_gsl(PyObject *, PyObject *);

    } // of namespace extensions
} // of namespace altar

#endif

// end of file
