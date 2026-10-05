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

// the panel width and the tile size of the blocked factorization
constexpr int LU_PANEL = 32;
constexpr int LU_TILE = 32;

// the shared memory the trailing updates stage their tiles in, one buffer for the real and the
// complex factorizations alike, sized for the complex ones
__device__ __forceinline__ unsigned char * lu_tiles()
{
    __shared__ alignas(16) unsigned char tiles[2*LU_TILE*(LU_PANEL + 1)*2*sizeof(double)];
    return tiles;
}

// factor A = P L U in place, LAPACK style: L (unit diagonal) below, U on and above the diagonal,
// {piv[k]} the row swapped with row k; returns whether A is singular; blocked, right looking: a
// panel of LU_PANEL columns at a time, whose trailing update goes through shared memory in tiles
template <class T, class S>
__device__ bool lu_factor(const cg::thread_block & cta, S * A, int * piv, const int n)
{
    __shared__ bool singular;
    if (cta.thread_rank() == 0)
        singular = false;
    cta.sync();
    auto tid = static_cast<int>(cta.thread_rank());
    auto threads = static_cast<int>(cta.size());
    // the tiles of L and U, padded against bank conflicts
    auto Lt = reinterpret_cast<S *>(lu_tiles());
    auto Ut = Lt + LU_TILE*(LU_PANEL + 1);
    for (auto k0 = 0; k0 < n; k0 += LU_PANEL) {
        auto kend = min(k0 + LU_PANEL, n);
        // factor the panel, column by column
        for (auto k = k0; k < kend; k++) {
            // pick the pivot, and swap it into row k, across the whole row
            auto p = pivot_row<T, S>(cta, A, n, k);
            if (tid == 0)
                piv[k] = p;
            if (p != k) {
                for (auto j = tid; j < n; j += threads) {
                    auto a = A[k*n + j];
                    A[k*n + j] = A[p*n + j];
                    A[p*n + j] = a;
                }
            }
            cta.sync();
            auto akk = A[k*n + k];
            if (magnitude(akk) == static_cast<T>(0)) {
                if (tid == 0)
                    singular = true;
                cta.sync();
                continue;
            }
            // the multipliers
            for (auto i = k + 1 + tid; i < n; i += threads)
                A[i*n + k] /= akk;
            cta.sync();
            // and the update of the rest of the panel
            auto w = kend - k - 1;
            auto m = n - k - 1;
            for (auto idx = tid; idx < m*w; idx += threads) {
                auto i = k + 1 + idx / w;
                auto j = k + 1 + idx % w;
                A[i*n + j] -= A[i*n + k]*A[k*n + j];
            }
            cta.sync();
        }
        // nothing left to the right of the last panel
        if (kend == n)
            break;
        // the rows of U to the right of the panel: each thread solves for its own columns with
        // the unit lower triangle of the panel
        for (auto j = kend + tid; j < n; j += threads) {
            for (auto r = k0 + 1; r < kend; r++) {
                auto acc = A[r*n + j];
                for (auto q = k0; q < r; q++)
                    acc -= A[r*n + q]*A[q*n + j];
                A[r*n + j] = acc;
            }
        }
        cta.sync();
        // the trailing update A22 -= L21 U12, a tile at a time
        auto nb = kend - k0;
        auto m = n - kend;
        auto tiles = (m + LU_TILE - 1) / LU_TILE;
        for (auto tile = 0; tile < tiles*tiles; tile++) {
            auto i0 = kend + (tile / tiles)*LU_TILE;
            auto j0 = kend + (tile % tiles)*LU_TILE;
            // stage the tiles of L21 and U12
            for (auto e = tid; e < LU_TILE*nb; e += threads) {
                auto r = e / nb;
                auto q = e % nb;
                Lt[r*(LU_PANEL + 1) + q] = (i0 + r < n) ? A[(i0 + r)*n + k0 + q] : S(0);
            }
            for (auto e = tid; e < nb*LU_TILE; e += threads) {
                auto q = e / LU_TILE;
                auto c = e % LU_TILE;
                Ut[q*(LU_TILE + 1) + c] = (j0 + c < n) ? A[(k0 + q)*n + j0 + c] : S(0);
            }
            cta.sync();
            // and update the tile of A22
            for (auto e = tid; e < LU_TILE*LU_TILE; e += threads) {
                auto r = e / LU_TILE;
                auto c = e % LU_TILE;
                if (i0 + r < n && j0 + c < n) {
                    auto acc = S(0);
                    for (auto q = 0; q < nb; q++)
                        acc += Lt[r*(LU_PANEL + 1) + q]*Ut[q*(LU_TILE + 1) + c];
                    A[(i0 + r)*n + j0 + c] -= acc;
                }
            }
            cta.sync();
        }
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
