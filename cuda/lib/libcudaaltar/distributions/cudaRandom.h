// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-2021 parasim inc
// (c) 2010-2021 california institute of technology
// all rights reserved
//
// Author(s): Hailiang Zhang, Lijun Zhu

/// altar/cuda/distributions/cudaRandom.h
/// template wrappers for curand generators

// code guard
#ifndef altar_cuda_distributions_cudaRandom_h
#define altar_cuda_distributions_cudaRandom_h

#include <curand_kernel.h>

// place everything in the local namespace
namespace altar {
    namespace cuda{
        namespace distributions {

            // wrappers for cuda random functions
            template <typename T>
            __device__ inline T curandUniform(curandState *state);

            template <typename T>
            __device__ inline T curandNormal(curandState *state);

            // inline functions
            template <>
            __device__ inline float curandUniform<float>(curandState *state) {
                return curand_uniform(state);
            }

            template <>
            __device__ inline double curandUniform<double>(curandState *state) {
                return curand_uniform_double(state);
            }

            // Specialization for float:
            template <>
            __device__ inline float curandNormal<float>(curandState *state) {
                return curand_normal(state);
            }

            // Specialization for double:
            template <>
            __device__ inline double curandNormal<double>(curandState *state) {
                return curand_normal_double(state);
            }

        } // of namespace distributions
    } // of namespace cuda
}// of namespace altar

#endif //altar_cuda_distributions_cudaRandom_h
// end of file
