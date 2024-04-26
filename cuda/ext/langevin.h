// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-2021 parasim inc
// (c) 2010-2021 california institute of technology
// all rights reserved
//
// Author(s): Lijun Zhu


#if !defined(cualtar_extensions_langevin_h)
#define cualtar_extensions_langevin_h


// place everything in my private namespace
namespace altar{ namespace cuda { namespace extensions { 
        // langevin      
        namespace cudaLangevin {
              
            extern const char * const updateTheta__name__;
            extern const char * const updateTheta__doc__;
            PyObject * updateTheta(PyObject *, PyObject *);
            
        } 
} } } // of namespace altar.cuda.extensions

#endif

// end of file
