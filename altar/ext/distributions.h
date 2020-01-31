// -*- C++ -*-
//
// Lijun Zhu (ljzhu@caltech.edu)
//
// (c) 2013-2020 parasim inc
// (c) 2010-2020 california institute of technology
// all rights reserved
//

#if !defined(altar_extensions_distributions_h)
#define altar_extensions_distributions_h


// place everything in my private namespace
namespace altar {
    namespace extensions {

        // for all ranged distribution
        namespace distribution {
            // verify whether parameters within the given range
            extern const char * const verify__name__;
            extern const char * const verify__doc__;
            PyObject * verify(PyObject *, PyObject *);
        }

        // uniform distribution        
        namespace uniform {
            extern const char * const sample__name__;
            extern const char * const sample__doc__;
            PyObject * sample(PyObject *, PyObject *);

            extern const char * const logpdf__name__;
            extern const char * const logpdf__doc__;
            PyObject * logpdf(PyObject *, PyObject *);

        } // of namespace uniform

        // gaussian distribution
        namespace gaussian {
            extern const char * const sample__name__;
            extern const char * const sample__doc__;
            PyObject * sample(PyObject *, PyObject *);

            extern const char * const logpdf__name__;
            extern const char * const logpdf__doc__;
            PyObject * logpdf(PyObject *, PyObject *);
        } //of namespace gaussian

    } // of namespace extensions
} // of namespace altar

#endif

// end of file
