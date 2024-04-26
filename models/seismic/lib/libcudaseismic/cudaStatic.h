// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-2021 parasim inc
// (c) 2010-2021 california institute of technology
// all rights reserved
//
// Author(s): Lijun Zhu

// code guard
#if !defined(altar_models_seismic_cudaStatic_h)
#define altar_models_seismic_cudaStatic_h



// macros
#include <cublas_v2.h>

// place everything in the local namespace
namespace altar {
    namespace models {
        namespace seismic {
            namespace cudaStatic {
            // forward declarations
                template <typename TYPE>
                void gemm_col(
                    const TYPE * const A, // M*K
                    const TYPE * const B, // K*N
                    TYPE * const C,      // M*1
                    const int M, const int K, const int N,
                    const int index, const TYPE factor, cudaStream_t stream=0);
            }
        } // of namespace seismic
    } // of namespace models
} // of namespace altar



#endif
