// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2022 - 2023 california institute of technology
// all rights reserved

// code guard
#ifndef __CUBLAS_CXX_WRAPPERS__
#define __CUBLAS_CXX_WRAPPERS__

#include <cublas_v2.h>
#include <iostream>

// gemm c++ template wrapper
template <class T>
cublasStatus_t cuBLASGemm(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m, int n, int k,
    const T* alpha,
    const T* A, int lda,
    const T* B, int ldb,
    const T* beta,
    T* C, int ldc
) {
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

// Specialization for float
template <>
cublasStatus_t cuBLASGemm<float>(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m, int n, int k,
    const float* alpha,
    const float* A, int lda,
    const float* B, int ldb,
    const float* beta,
    float* C, int ldc
) {
    return cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

// Specialization for double
template <>
cublasStatus_t cuBLASGemm<double>(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m, int n, int k,
    const double* alpha,
    const double* A, int lda,
    const double* B, int ldb,
    const double* beta,
    double* C, int ldc
) {
    return cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

// Specialization for cuComplex (float complex)
template <>
cublasStatus_t cuBLASGemm<cuComplex>(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m, int n, int k,
    const cuComplex* alpha,
    const cuComplex* A, int lda,
    const cuComplex* B, int ldb,
    const cuComplex* beta,
    cuComplex* C, int ldc
) {
    return cublasCgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

// Specialization for cuDoubleComplex (double complex)
template <>
cublasStatus_t cuBLASGemm<cuDoubleComplex>(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m, int n, int k,
    const cuDoubleComplex* alpha,
    const cuDoubleComplex* A, int lda,
    const cuDoubleComplex* B, int ldb,
    const cuDoubleComplex* beta,
    cuDoubleComplex* C, int ldc
)
{
    return cublasZgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}


template <typename T>
cublasStatus_t cuBLASGemmStridedBatched(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m, int n, int k,
    const T* alpha,
    const T* A, int lda, long long int strideA,
    const T* B, int ldb, long long int strideB,
    const T* beta,
    T* C, int ldc, long long int strideC,
    int batchCount
) {
    std::cout << "gemm batched only defined for float and double\n";
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

template <>
cublasStatus_t cuBLASGemmStridedBatched<float>(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m, int n, int k,
    const float* alpha,
    const float* A, int lda, long long int strideA,
    const float* B, int ldb, long long int strideB,
    const float* beta,
    float* C, int ldc, long long int strideC,
    int batchCount
) {
    std::cout << "gemm float version called with" << batchCount << " batches \n";
    cublasStatus_t status = cublasSgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
    return status;
}

template <>
cublasStatus_t cuBLASGemmStridedBatched<double>(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m, int n, int k,
    const double* alpha,
    const double* A, int lda, long long int strideA,
    const double* B, int ldb, long long int strideB,
    const double* beta,
    double* C, int ldc, long long int strideC,
    int batchCount
) {
    std::cout << "gemm double version called with" << batchCount << " batches \n";
    cublasStatus_t status = cublasDgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
    return status;
   }

// more to be added when needed

#endif // __CUBLAS_CXX_WRAPPERS__
// end of file
