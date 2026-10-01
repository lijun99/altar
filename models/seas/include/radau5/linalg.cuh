// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2026 california institute of technology
// all rights reserved

/**
 * linalg.cuh
 * dense linear algebra by a thread block, for one system: LU with partial pivoting and its solve,
 * for real or complex (cuda::std::complex) matrices in row-major global memory
 **/

// code guard
#ifndef cuda_ode_radau5_linalg_cuh
#define cuda_ode_radau5_linalg_cuh

#include <dopri5/external.h>
#include <cuda/std/complex>

namespace cuda::ode::radau5 {

// the magnitude used to pick pivots; |re| + |im| for complex numbers
template <class T>
__device__ __forceinline__ T magnitude(const T x) { return abs(x); }

template <class T>
__device__ __forceinline__ T magnitude(const cuda::std::complex<T> x)
{
    return abs(x.real()) + abs(x.imag());
}

// the row i >= k with the largest |A[i, k]|, shared by all threads of the block
template <class T, class S>
__device__ int pivot_row(const cg::thread_block & cta, const S * A, const int n, const int k)
{
    __shared__ T warp_best[32];
    __shared__ int warp_index[32];
    __shared__ int result;

    // the best candidate of each thread, ties going to the lower row
    auto best = static_cast<T>(-1);
    auto index = k;
    for (auto i = k + static_cast<int>(cta.thread_rank()); i < n; i += cta.size()) {
        auto m = magnitude(A[i*n + k]);
        if (m > best) {
            best = m;
            index = i;
        }
    }

    // reduce within each warp
    auto tile = cg::tiled_partition<32>(cta);
    for (int offset = 16; offset > 0; offset /= 2) {
        auto other_best = tile.shfl_down(best, offset);
        auto other_index = tile.shfl_down(index, offset);
        if (other_best > best || (other_best == best && other_index < index)) {
            best = other_best;
            index = other_index;
        }
    }
    auto warp = cta.thread_rank() / 32;
    if (tile.thread_rank() == 0) {
        warp_best[warp] = best;
        warp_index[warp] = index;
    }
    cta.sync();

    // and across the warps
    if (cta.thread_rank() == 0) {
        auto warps = (cta.size() + 31) / 32;
        auto b = warp_best[0];
        auto j = warp_index[0];
        for (auto w = 1u; w < warps; w++) {
            if (warp_best[w] > b || (warp_best[w] == b && warp_index[w] < j)) {
                b = warp_best[w];
                j = warp_index[w];
            }
        }
        result = j;
    }
    cta.sync();
    return result;
}

// factor A = P L U in place, LAPACK style: L (unit diagonal) below, U on and above the diagonal,
// {piv[k]} the row swapped with row k; returns whether A is singular
template <class T, class S>
__device__ bool lu_factor(const cg::thread_block & cta, S * A, int * piv, const int n)
{
    __shared__ bool singular;
    if (cta.thread_rank() == 0)
        singular = false;
    cta.sync();

    for (auto k = 0; k < n; k++) {
        // pick the pivot, and swap it into row k
        auto p = pivot_row<T, S>(cta, A, n, k);
        if (cta.thread_rank() == 0)
            piv[k] = p;
        if (p != k) {
            for (auto j = static_cast<int>(cta.thread_rank()); j < n; j += cta.size()) {
                auto a = A[k*n + j];
                A[k*n + j] = A[p*n + j];
                A[p*n + j] = a;
            }
        }
        cta.sync();

        auto akk = A[k*n + k];
        if (magnitude(akk) == static_cast<T>(0)) {
            if (cta.thread_rank() == 0)
                singular = true;
            cta.sync();
            continue;
        }

        // the multipliers
        for (auto i = k + 1 + static_cast<int>(cta.thread_rank()); i < n; i += cta.size())
            A[i*n + k] /= akk;
        cta.sync();

        // and the update of the trailing block
        auto m = n - k - 1;
        for (auto idx = static_cast<int>(cta.thread_rank()); idx < m*m; idx += cta.size()) {
            auto i = k + 1 + idx / m;
            auto j = k + 1 + idx % m;
            A[i*n + j] -= A[i*n + k]*A[k*n + j];
        }
        cta.sync();
    }
    return singular;
}

// solve A x = b in place, with A factored by {lu_factor}
template <class T, class S>
__device__ void lu_solve(const cg::thread_block & cta, const S * LU, const int * piv, S * b, const int n)
{
    // apply the row swaps, in order
    if (cta.thread_rank() == 0) {
        for (auto k = 0; k < n; k++) {
            auto p = piv[k];
            if (p != k) {
                auto t = b[k];
                b[k] = b[p];
                b[p] = t;
            }
        }
    }
    cta.sync();

    // forward substitution with the unit lower triangle
    for (auto k = 0; k < n - 1; k++) {
        auto bk = b[k];
        for (auto i = k + 1 + static_cast<int>(cta.thread_rank()); i < n; i += cta.size())
            b[i] -= LU[i*n + k]*bk;
        cta.sync();
    }

    // back substitution with the upper triangle
    for (auto k = n - 1; k >= 0; k--) {
        if (cta.thread_rank() == 0)
            b[k] /= LU[k*n + k];
        cta.sync();
        auto bk = b[k];
        for (auto i = static_cast<int>(cta.thread_rank()); i < k; i += cta.size())
            b[i] -= LU[i*n + k]*bk;
        cta.sync();
    }
}

} // end of namespace cuda::ode::radau5

#endif // cuda_ode_radau5_linalg_cuh
// end of file
