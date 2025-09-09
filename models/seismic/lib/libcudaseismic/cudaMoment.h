// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-2021 parasim inc
// (c) 2010-2021 california institute of technology
// all rights reserved
//
// Author(s): Lijun Zhu

/// altar/cuda/distributions/cudaMoment.h
/// Uniform Distribution

// code guard
#pragma once

// dependencies
#include <cuda_runtime.h>

// place everything in the local namespace
// place everything in the local namespace
namespace altar {
    namespace models {
        namespace seismic {
            namespace cudaMoment {

                // calculate log probability
                template <typename real_type>
                void logpdf(const real_type * const theta, real_type * const probability, 
                    const size_t samples, const size_t parameters, 
                    const size_t idx_begin, const size_t idx_end,
                    const real_type mean, const real_type sigma, 
                    const real_type * const mu_area, const real_type moment_constraint_factor,
                    cudaStream_t stream=0);

            } // of namespace cudaMoment
        } // of namespace seismic
    } // of namespace models
} // of namespace altar

// end of file
