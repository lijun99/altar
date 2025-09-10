// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-2025 parasim inc
// (c) 2010-2025 california institute of technology
// all rights reserved
//
// Author(s): Lijun Zhu

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
