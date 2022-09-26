// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2022 california institute of technology
// all rights reserved
//

// code guard
#ifndef altar_model_seas_cuda_details_cuh
#define altar_model_seas_cuda_details_cuh

// my dependencies
#include <pyre/cuda.h>

namespace details {

// print out a segment of cuda memory for debugging purposes
template<typename T>
void debug_cuda_memory(const T* mem, const int size)
{
    T * h_mem = (T *) malloc(size*sizeof(T));
    cudaSafeCall(cudaMemcpy(h_mem, mem, size*sizeof(T), cudaMemcpyDeviceToHost));
    int out = std::min(16, size);
    for(int i=0; i<out; ++i)
        std::cout << h_mem[i] << " ";
    std::cout << "...\n";
    for(int i=size-out; i<size; i++)
        std::cout << h_mem[i] << " ";
    std::cout << "\n";
    free(h_mem);
}


template <typename T>
void matrix_copy(T* dst, const T* src, const int size)
{
    cudaSafeCall(cudaMemcpy(dst, src, size*sizeof(T), cudaMemcpyDeviceToDevice));
}

// add vector to rows of a matrix - kernel
template<typename T>
__global__ void matrix_add_vector_to_rows_kernel(T* matrix, const T* vector, const int rows, const int cols)
{
    // each row uses a thread, get row index from thread id
    int row  = blockIdx.x *blockDim.x + threadIdx.x;
    // avoid out of range
    if (row>=rows)
        return;
    // get the head of the matrix row
    auto mat_row = matrix + row*cols;
    // iterate over col
    for(int i=0; i<cols; ++i)
        mat_row[i] += vector[i];
    // all done
}

// add vector to rows of a matrix - host function
// matrix = matrix + vector
// @param matrix (rows, cols)
// @param vector (cols)
template <typename T>
void matrix_add_vector_to_rows(T* matrix, const T* vector, const int rows, const int cols)
{
    // decide the execution size - one row per thread
    const int threadsPerBlock = 128;
    const int numberOfBlocks = (rows-1+threadsPerBlock)/threadsPerBlock; //IDIVUP
    // call kernel
    matrix_add_vector_to_rows_kernel<T><<<numberOfBlocks, threadsPerBlock>>>(matrix, vector, rows, cols);
    // check error
   cudaSafeCall(cudaGetLastError());
}

// add vector to rows of a matrix - kernel
template<typename T>
__global__ void matrix_add_vector_to_rows_kernel2(T* matrix, const T* matrix0, const T* vector, const int rows, const int cols)
{
    // each row uses a thread, get row index from thread id
    int row  = blockIdx.x *blockDim.x + threadIdx.x;
    // avoid out of range
    if (row>=rows)
        return;
    // get the head of the matrix row
    auto mat_row = matrix + row*cols;
    auto mat0_row = matrix0 + row*cols;
    // iterate over col
    for(int i=0; i<cols; ++i)
        mat_row[i] = mat0_row[i] + vector[i];
    // all done
}

// add vector to rows of a matrix - host function
// matrix = matrix0 + vector
// @param matrix (rows, cols)
// @param vector (cols)
template <typename T>
void matrix_add_vector_to_rows(T* matrix, const T* matrix0, const T* vector, const int rows, const int cols)
{
    // decide the execution size - one row per thread
    const int threadsPerBlock = 128;
    const int numberOfBlocks = (rows-1+threadsPerBlock)/threadsPerBlock; //IDIVUP
    // call kernel
    matrix_add_vector_to_rows_kernel2<T><<<numberOfBlocks, threadsPerBlock>>>(matrix, matrix0, vector, rows, cols);
    // check error
    cudaSafeCall(cudaGetLastError());
}

// duplicate vector to rows of a matrix - kernel
template<typename T>
__global__ void matrix_duplicate_vector_kernel(T* matrix, const T* vector, const int rows, const int cols)
{
    // each row uses a thread, get row index from thread id
    int row  = blockIdx.x *blockDim.x + threadIdx.x;
    // avoid out of range
    if (row>=rows)
        return;
    // get the head of the matrix row
    auto mat_row = matrix + row*cols;
    // iterate over col
    for(int i=0; i<cols; ++i)
        mat_row[i] = vector[i];
    // all done
}

// add vector to rows of a matrix - host function
// matrix = matrix + vector
// @param matrix (rows, cols)
// @param vector (cols)
template <typename T>
void matrix_duplicate_vector(T* matrix, const T* vector, const int rows, const int cols)
{
    // decide the execution size - one row per thread
    const int threadsPerBlock = 128;
    const int numberOfBlocks = (rows-1+threadsPerBlock)/threadsPerBlock; //IDIVUP
    // call kernel
    matrix_duplicate_vector_kernel<T><<<numberOfBlocks, threadsPerBlock>>>(matrix, vector, rows, cols);
    // check error
   cudaSafeCall(cudaGetLastError());
}

template <typename T>
T vector_sum(const T * vector, const int elements)
{
    T * h_vector = (T *) malloc(elements*sizeof(T));
    cudaSafeCall(cudaMemcpy(h_vector, vector, elements*sizeof(T), cudaMemcpyDeviceToHost));

    T result = (T)0;
    for(int i=0; i<elements; i++)
        result += h_vector[i];
    free(h_vector);
    return result;
}

} // end of namespace
#endif
// end of file
