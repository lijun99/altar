// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2022 - 2023 california institute of technology
// all rights reserved
//

// This file includes routines to perform d= G x M in the forward model,
// shared by differet_steps rheology problems

// code guard
#if !defined(altar_models_seas_cuda_displacemet_steps_cuh)
#define altar_models_seas_cuda_displacemet_steps_cuh

namespace altar::models::seas::cuda::displacemet_steps {

// Suppose yeval is arranged in [systems, neval/t_steps, systems_size]
// in order to use batched gemm, we need to reshape yeval into [neval/t_steps, systems, slip_size]

template <typename T>
__global__ void slip_reformat_kernel(T* slips, const T* y,
    const int systems,
    const int t_steps,
    const int system_size,
    const int slip_size)
{
    // thread blocks along x - systems*slip_size
    auto thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    // check range
    if (thread_id >= system_size*slip_size)
        return;
    auto sys_id = thread_id / slip_size;
    auto slip_id = thread_id % slip_size;
    // thread blocks along y - t_steps
    auto time_id = blockIdx.y;

    // flatten the output index
    auto dst_id = (time_id * systems + sys_id) * slip_size + slip_id;
    // flatten the input index
    auto src_id = (sys_id * t_steps + time_id) * system_size + slip_id;
    // copy over
    slips[dst_id] = y[src_id];
    // all done
}


template <typename T>
void slip_reformat(T* slips, const T* y,
    const int systems,
    const int t_steps,
    const int system_size,
    const int slip_size)
{
    auto total_threads = systems*slip_size;

    // get the number of threads (along x)
    int threads;

    if(total_threads <= 32)
        threads = 32;
    else if (total_threads <= 64)
        threads = 64;
    else if (total_threads <= 128)
        threads =128;
    else if (total_threads <= 256)
        threads = 256;
    else if (total_threads <=512 )
        threads = 512;
    else
        threads = 1024;
    // get the number of blocks
    auto blocks = (total_threads + threads-1)/threads;

    // the kernel execution plan
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, t_steps, 1);

    // call the kernel
    slip_reformat_kernel<T><<<dimGrid, dimBlock>>>(slips, y, systems, t_steps, system_size, slip_size);
    cudaCheckError("slip_reformat_kernel error");
    // all done
}


// The displacement used for batched gemm is in shape [tsteps, samples, stations]
// We need to reformat it to [samples, tsteps, stations] for residue and likelihood computation


template <typename T>
__global__ void displacement_reshape_kernel(T* disp_out, const T* disp_in,
    const int systems,
    const int t_steps,
    const int stations)
{

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int index1, index2;
    if (i < m && j < n) {
        for (int l = 0; l < k; l++) {
            index1 = i * n * k + j * k + l;
            index2 = j * m * k + i * k + l;
            B[index2] = A[index1];
        }
    }
    // all done
}


// overload float/double precisions gemmStridedBatched
cublasStatus_t cublasGemmStridedBatched(cublasHandle_t handle,
                                  cublasOperation_t transa,
                                  cublasOperation_t transb,
                                  int m, int n, int k,
                                  const float           *alpha,
                                  const float           *A, int lda,
                                  long long int          strideA,
                                  const float           *B, int ldb,
                                  long long int          strideB,
                                  const float           *beta,
                                  float                 *C, int ldc,
                                  long long int          strideC,
                                  int batchCount)
{
    return cublasSgemmStridedBatched(handle, transa, transb,
        m, n, k,
        alpha,
        A, lda, strideA,
        B, ldb, strideB,
        beta,
        C, ldc, strideC,
        batchCount);
}

cublasStatus_t cublasGemmStridedBatched(cublasHandle_t handle,
                                  cublasOperation_t transa,
                                  cublasOperation_t transb,
                                  int m, int n, int k,
                                  const double          *alpha,
                                  const double          *A, int lda,
                                  long long int          strideA,
                                  const double          *B, int ldb,
                                  long long int          strideB,
                                  const double          *beta,
                                  double                *C, int ldc,
                                  long long int          strideC,
                                  int batchCount)
{
    return cublasDgemmStridedBatched(handle, transa, transb,
        m, n, k,
        alpha,
        A, lda, strideA,
        B, ldb, strideB,
        beta,
        C, ldc, strideC,
        batchCount);
}



// @param gf [t_steps, slip_size, stations]

template <typename T>
void compute_displacemet(cublasHandle_t handle, T* predictions, const T* yeval, const T* gf,
        const int samples, // samples or systems
        const int t_steps, const int system_size, const int slip_size,
        const int stations)
{

    T* slips;

    cudaSafeCall(cudaMalloc(&slips, systems*t_steps*slip_size*sizeof(T));

    // reformat/extract slip or slip rate from yeval
    slip_reformat(slips, yeval, samples, t_steps, system_size, slip_size);

    // Call Gemm Batched C = alph A B + beta C
    // slip [samples, slip_size] * G [slip_size, stations] = displacement [samples, stations]
    // cublas use Column-major, which is equivalent to
    //  G [stations, slip_size] * slip[slip_size, samples] = displacement [stations, samples]
    cublasSafeCall( cublasGemmStridedBatched(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        stations, samples, slip_size,
        (T)1.0,
        gf, stations, stations*slip_size,
        slips, slip_size, slip_size*samples,
        (T)0.0,
        predictions, stations, stations*samples,
        t_steps);
    ));

}


} // end of namespace

#endif
// end of file
