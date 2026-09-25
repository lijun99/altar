// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//

// code guard
#pragma once

// place everything in my private namespace
namespace altar {
    namespace extensions {
        namespace models {
            namespace cudaseismic {

                // compute log probability
                extern const char * const moment_logpdf__name__;
                extern const char * const moment_logpdf__doc__;
                PyObject * moment_logpdf(PyObject *, PyObject *);
                
            } // of namespace cudaseismic
        } // of namespace models
    } // of namespace extensions
} // of namespace altar

// end of file
