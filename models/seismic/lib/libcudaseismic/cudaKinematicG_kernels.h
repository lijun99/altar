// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-2019 parasim inc
// (c) 2010-2019 california institute of technology
// all rights reserved
//
// Author(s): Hailiang Zhang, Lijun Zhu

#ifndef cudaKinematicG_kernels_h
#define cudaKinematicG_kernels_h

// macros
#include <cuda.h>
#include <cuda_runtime.h>

// place everything in the local namespace

/// @brief home of cuda kernel for kinematic model with big-G implementation
namespace cudaKinematicG_kernels {
    /// Initialize T0 to be distances!!!
    //single sample
    template<typename TYPE>
    __device__ void initT0(TYPE * const gT0, const size_t Nddf, const size_t Nasf,
        TYPE dspf, TYPE hypo_dip, TYPE hypo_strike, TYPE it0);
    // batched samples
    template <typename TYPE>
    __global__ void initT0_batched(const size_t * const gIdx, const TYPE *const gM, TYPE * const gT0, const size_t Nparam,
        const size_t Nas, const size_t Ndd, const size_t Nmesh, const TYPE dsp, const TYPE it0);
    
    /// Set t0 for the 4 patches closest to hypo center; Set all other T0s to be a large number (it0)
    // single sample
    template <typename TYPE>
    __device__ void setT0hypo(const size_t * gIdx, const TYPE *const gM, TYPE *const gT0,
        const size_t Nas, const size_t Ndd, const size_t Nmesh, const TYPE dsp, const TYPE it0);
    // batched samples
    template <typename TYPE>
    __global__ void setT0_batched(const size_t * gIdx, const TYPE *const gM, TYPE *const gT0, const size_t Nparam,
        const size_t Ns_good, const size_t Nas, const size_t Ndd, const size_t Nmesh, const TYPE dsp, const TYPE it0);

    /// Upwind on the device code
    // one sweep
    template <typename TYPE>
    __device__ TYPE upwind(const size_t * gIdx, const TYPE *const gT0, const int i, const int j,
        const size_t Nas, const size_t Ndd, const size_t Nmesh, const TYPE *const gM, const TYPE h);
    /// Fast Sweeping
    template <typename TYPE>
    __global__ void fastSweeping_batched(const size_t * gIdx, const TYPE *const gM, TYPE *const gT0,
        const size_t Nparam, const size_t Ns_good, const size_t Nas, const size_t Ndd, const size_t Nmesh,
        const TYPE h, const size_t iteration);

    /// @brief size_terpolate T0 to TI0
    /// A coordinate system is constructed with gT0[0] as the origin, and each patch length as 1.0<br>
    /// The leading index is along Ndd direction
    template <typename TYPE>
    __global__ void interpolateT0_batched(const TYPE *const gT0, TYPE *const gTI0, const size_t Ns_good,
        const size_t Nas, const size_t Ndd, const size_t Nmesh, const size_t Npt_gi);

    /// @brief Cast M to M_big
    /// @note The leading indices are both Ns (number of samples)
    template <typename TYPE>
    __global__ void castBigM_batched(const size_t * gIdx, const TYPE *const gM, const TYPE *const gTI0,
        TYPE *const gMb, const TYPE *const gt0s,
        const TYPE dt, const size_t Nparam,
        const size_t Nt, const size_t Nas, const size_t Ndd, const size_t Npt_gi);
    
} // of namespace cudaKinematicsG_kernerls

#endif
//end of file
