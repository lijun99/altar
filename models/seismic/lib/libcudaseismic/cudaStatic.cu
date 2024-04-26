// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-2021 parasim inc
// (c) 2010-2021 california institute of technology
// all rights reserved
//
// Author(s): Lijun Zhu

// for the build system
#include <portinfo>

// get my class declaration
#include "cudaStatic.h"

// my dependencies
#include <pyre/cuda.h>
#include <algorithm>
#include <iostream>


namespace cudaStatic_kernels {

    template<typename TYPE>
    __global__ void gemm_col_kernel(
        const TYPE * const A, // M*K
        const TYPE * const B, // K*N
        TYPE * const C,      // M*1
        const int M, const int K, const int N,
        const int i, const TYPE factor);

} // of namespace cudaStatic_kernels


template <typename TYPE>
void
altar::models::seismic::cudaStatic::
gemm_col(
    const TYPE * const A, // M*K
    const TYPE * const B, // K*N
    TYPE * const C,      // M*1
    const int M, const int K, const int N,
    const int index, const TYPE factor, cudaStream_t stream)
{
    int blockSize = NTHREADS;
    int gridSize = IDIVUP(M, blockSize);
    // call cuda kernels
    cudaStatic_kernels::gemm_col_kernel<TYPE><<<gridSize, blockSize, 0, stream>>>(
        A, B, C, M, K, N, index, factor);
    cudaCheckError("cudaStatic:gemm_col error");
    // all done
}

// explicit instantiation
template void altar::models::seismic::cudaStatic::gemm_col<float>(const float * const, const float * const,
    float * const, const int, const int, const int, const int, const float, cudaStream_t);
template void altar::models::seismic::cudaStatic::gemm_col<double>(const double * const, const double * const,
    double * const, const int, const int, const int, const int, const double, cudaStream_t);

template<typename TYPE>
__global__ void
cudaStatic_kernels::
gemm_col_kernel(
        const TYPE * const A, // M*K
        const TYPE * const B, // K*N
        TYPE * const C,      // M*1
        const int M, const int K, const int N,
        const int index, const TYPE factor)
{
    int m = threadIdx.x + blockIdx.x * blockDim.x;
    if(m >= M) return;

    TYPE sum = 0;
    for(int k=0; k< K; ++k)
        sum += A[m*K+k]*B[k*N+index];
    C[m] = factor*sum;
    // all done
    return;
}


// end of file
