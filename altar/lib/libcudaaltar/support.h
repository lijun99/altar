// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//

//! shared support for altar's cuda kernels: a default thread block size, a ceiling-division
//! helper for turning an element count into a grid size, and a way to check the outcome of a
//! kernel launch -- the same shape as pyre's own {pyre::py::cuda::checkCuda}, just without a
//! python exception at the end of it, since this runs on the host side of a kernel launch,
//! not behind a pybind11 binding

// code guard
#ifndef altar_cuda_support_h
#define altar_cuda_support_h

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <string>

// pyre's own statically typed grid; {WITH_CUDA} (set for the whole {libcudaaltar} target,
// see altar_cudaframework.cmake) turns on the {PYRE_HOST_DEVICE} decorations {matrix_view_t}/
// {vector_view_t} need to be indexable from a kernel; {--expt-relaxed-constexpr} (also set
// there) lets device code reach the constexpr machinery under {Shape}/{Index}
#include <pyre/grid.h>
#include <pyre/memory.h>

// global scope, deliberately: the kernel files that use these nest themselves differently --
// some inside {altar::cuda::bayesian::cudaLeapfrog}, some in a bare top-level namespace of
// their own -- and the macros these replace were global by nature, so matching that keeps
// every call site's unqualified {NTHREADS}/{IDIVUP(...)}/{cudaCheckError(...)} working
// regardless of which namespace it's nested in

// the default number of threads per block; a kernel launch is free to pick its own instead
constexpr int NTHREADS = 256;

// the number of blocks of {threads} needed to cover {elements}
constexpr auto
IDIVUP(int elements, int threads) -> int
{
    return (elements + threads - 1) / threads;
}

// check the outcome of the most recent kernel launch, exactly like {cudaCheckError(msg)}
// used to; unlike the macro it replaces, this doesn't reach for a bespoke exception
// hierarchy from a package that no longer exists -- a plain runtime_error carrying cuda's
// own message is all a kernel launch failure needs
inline auto
cudaCheckError(const char * msg) -> void
{
    auto status = cudaGetLastError();
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(status));
    }
}

// check the outcome of a cuda runtime call directly, e.g. {cudaSafeCall(cudaMalloc(...))};
// the same idea as {cudaCheckError} above, just wrapping the call's own return value
// instead of asking the runtime what the last error was
inline auto
cudaSafeCall(cudaError_t status) -> void
{
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

// the cublas counterpart of {cudaSafeCall}, e.g. {cublasSafeCall(cublasSgemm(handle, ...))};
// {cublasGetStatusString} gives a human readable message the same way {cudaGetErrorString}
// does for the plain cuda runtime
inline auto
cublasSafeCall(cublasStatus_t status) -> void
{
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(cublasGetStatusString(status));
    }
}

// the default number of threads per block for kernels that don't otherwise pick their own;
// the old {altar/utils/common.h} macro of the same name is gone along with the rest of the
// pre-pybind11 headers, but the concept (and value) it named lives on here, as a plain
// {constexpr} alongside {NTHREADS}
constexpr int BLOCKDIM = 256;

// pi, for kernels that need it and don't want to depend on {M_PI} being defined (it isn't,
// on every platform, without {_USE_MATH_DEFINES})
constexpr double PI = 3.14159265358979323846;

// the 2d thread block shape a kernel launch over (samples, parameters) tiles with; 32 along
// the sample axis keeps memory access coalesced across a warp, 4 along the parameter axis
// keeps the total at a full 128 threads -- a plain, portable default, not a tuned one
constexpr int BDIMX = 32;
constexpr int BDIMY = 4;

// the linear offset of {(row, col)} in a row-major matrix with {cols} columns; called from
// both host launch-configuration code and device kernel bodies, hence both annotations
__host__ __device__ constexpr auto
IDX2R(int row, int col, int cols) -> int
{
    return row * cols + col;
}

// a non-owning (samples x parameters) view over someone else's cells, the shape every
// kernel in this library that operates on a batch of samples takes its data through -- reuse
// this instead of hand-rolling the same {pyre::grid::grid_t<...>} spelling per file. {isConst}
// defaults to {true} (read-only), the common case; pass {false} for a view a kernel writes
// into (e.g. {sample}'s {theta}, or a full gradient matrix)
template <typename real_type, bool isConst = true>
using matrix_view_t = pyre::grid::grid_t<
    pyre::grid::canonical_t<2>, pyre::memory::View<real_type, isConst>>;

// the (samples,) counterpart of {matrix_view_t}, e.g. a per-sample probability or log
// likelihood; {isConst} defaults to {false} here instead, since a (samples,) view is more
// often something a kernel fills in than something it only reads
template <typename real_type, bool isConst = false>
using vector_view_t = pyre::grid::grid_t<
    pyre::grid::canonical_t<1>, pyre::memory::View<real_type, isConst>>;

#endif

// end of file
