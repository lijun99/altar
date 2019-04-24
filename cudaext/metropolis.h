// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-2019 parasim inc
// (c) 2010-2019 california institute of technology
// all rights reserved
//
// Author(s): Lijun Zhu


#if !defined(cualtar_extensions_metropolis_h)
#define cualtar_extensions_metropolis_h


// place everything in my private namespace
namespace altar{ namespace cuda { namespace extensions { 
        // metropolis      
        namespace cudaMetropolis {
              
            extern const char * const setValidSampleIndices__name__;
            extern const char * const setValidSampleIndices__doc__;
            PyObject * setValidSampleIndices(PyObject *, PyObject *);
            
            extern const char * const queueValidSamples__name__;
            extern const char * const queueValidSamples__doc__;
            PyObject * queueValidSamples(PyObject *, PyObject *);

            extern const char * const metropolisUpdate__name__;
            extern const char * const metropolisUpdate__doc__;
            PyObject * metropolisUpdate(PyObject *, PyObject *);
            
        } 
} } } // of namespace altar.cuda.extensions

#endif

// end of file
