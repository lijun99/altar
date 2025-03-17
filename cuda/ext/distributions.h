// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-2021 parasim inc
// (c) 2010-2021 california institute of technology
// all rights reserved
//
// Author(s): Lijun Zhu


#if !defined(cualtar_extensions_distributions_h)
#define cualtar_extensions_distributions_h

#include "capsules.h"

// place everything in my private namespace
namespace altar { namespace cuda { namespace extensions {

        // ranged distribution
        namespace cudaRanged {
            // verify
            extern const char * const verify__name__;
            extern const char * const verify__doc__;
            PyObject * verify(PyObject *, PyObject *);

            extern const char * const verify_unique__name__;
            extern const char * const verify_unique__doc__;
            PyObject * verify_unique(PyObject *, PyObject *);


            // constrain
            extern const char * const constrain__name__;
            extern const char * const constrain__doc__;
            PyObject * constrain(PyObject *, PyObject *);
        }

        // uniform distribution
        namespace cudaUniform {
            // generate random sample
            extern const char * const sample__name__;
            extern const char * const sample__doc__;
            PyObject * sample(PyObject *, PyObject *);

            // compute log probability
            extern const char * const logpdf__name__;
            extern const char * const logpdf__doc__;
            PyObject * logpdf(PyObject *, PyObject *);

            // generate random sample
            extern const char * const sample_unique__name__;
            extern const char * const sample_unique__doc__;
            PyObject * sample_unique(PyObject *, PyObject *);

            // compute log probability
            extern const char * const logpdf_unique__name__;
            extern const char * const logpdf_unique__doc__;
            PyObject * logpdf_unique(PyObject *, PyObject *);

        }

         // logistic distribution, for other logit substitutions
        namespace cudaLogistic {
            // generate random sample
            extern const char * const sample__name__;
            extern const char * const sample__doc__;
            PyObject * sample(PyObject *, PyObject *);
            // compute log probability
            extern const char * const logpdf__name__;
            extern const char * const logpdf__doc__;
            PyObject * logpdf(PyObject *, PyObject *);
            // compute log probability gradient
            extern const char * const logpdfgradient__name__;
            extern const char * const logpdfgradient__doc__;
            PyObject * logpdfgradient(PyObject *, PyObject *);
        }

        // uniform distribution via a logit variable
        namespace cudaUniformLogit {
            // transform to sampling
            extern const char * const tosampling__name__;
            extern const char * const tosampling__doc__;
            PyObject * tosampling(PyObject *, PyObject *);
            // tranform to physical
            extern const char * const tophysical__name__;
            extern const char * const tophysical__doc__;
            PyObject * tophysical(PyObject *, PyObject *);
        }

        // gaussian distribution
        namespace cudaGaussian {
            // generate random sample
            extern const char * const sample__name__;
            extern const char * const sample__doc__;
            PyObject * sample(PyObject *, PyObject *);
            // compute log probability
            extern const char * const logpdf__name__;
            extern const char * const logpdf__doc__;
            PyObject * logpdf(PyObject *, PyObject *);
            // compute log probability gradient
            extern const char * const logpdfgradient_i__name__;
            extern const char * const logpdfgradient_i__doc__;
            PyObject * logpdfgradient_i(PyObject *, PyObject *);
            // compute log probability gradient
            extern const char * const logpdfgradient__name__;
            extern const char * const logpdfgradient__doc__;
            PyObject * logpdfgradient(PyObject *, PyObject *);
        }

        // truncated gaussian distribution
        namespace cudaTGaussian {
            // generate random sample
            extern const char * const sample__name__;
            extern const char * const sample__doc__;
            PyObject * sample(PyObject *, PyObject *);
            // compute log probability
            extern const char * const logpdf__name__;
            extern const char * const logpdf__doc__;
            PyObject * logpdf(PyObject *, PyObject *);
        }

        // truncated gaussian distribution via a logit variable
        namespace cudaTGaussianLogit {
            // transform to sampling
            extern const char * const tosampling__name__;
            extern const char * const tosampling__doc__;
            PyObject * tosampling(PyObject *, PyObject *);
            // tranform to physical
            extern const char * const tophysical__name__;
            extern const char * const tophysical__doc__;
            PyObject * tophysical(PyObject *, PyObject *);
        }

} } } // of namespace altar.cuda.extensions

#endif

// end of file
