// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//

#include <portinfo>

#include "cudaGaussianSlip.h"
#include "../libseismic/stf.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include <altar/cuda/support.h>

namespace {
    // the cusolver counterpart of {cudaSafeCall}
    void cusolverSafeCall(cusolverStatus_t status, const char * routine)
    {
        if (status != CUSOLVER_STATUS_SUCCESS) {
            throw std::runtime_error(std::string(routine) + ": cusolver status " + std::to_string(int(status)));
        }
    }

    // the batched blas and lapack of each precision
    cublasStatus_t gemmStridedBatched(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n,
        int k, const double * alpha, const double * A, int lda, long long sA, const double * B, int ldb,
        long long sB, const double * beta, double * C, int ldc, long long sC, int batch)
    {
        return cublasDgemmStridedBatched(h, ta, tb, m, n, k, alpha, A, lda, sA, B, ldb, sB, beta, C, ldc, sC, batch);
    }

    cublasStatus_t gemmStridedBatched(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n,
        int k, const float * alpha, const float * A, int lda, long long sA, const float * B, int ldb,
        long long sB, const float * beta, float * C, int ldc, long long sC, int batch)
    {
        return cublasSgemmStridedBatched(h, ta, tb, m, n, k, alpha, A, lda, sA, B, ldb, sB, beta, C, ldc, sC, batch);
    }

    cusolverStatus_t potrfBatched(cusolverDnHandle_t h, int n, double ** A, int * info, int batch)
    {
        return cusolverDnDpotrfBatched(h, CUBLAS_FILL_MODE_LOWER, n, A, n, info, batch);
    }

    cusolverStatus_t potrfBatched(cusolverDnHandle_t h, int n, float ** A, int * info, int batch)
    {
        return cusolverDnSpotrfBatched(h, CUBLAS_FILL_MODE_LOWER, n, A, n, info, batch);
    }

    // B <- L^-1 B or L^-T B, with L lower
    cublasStatus_t trsvBatched(cublasHandle_t h, cublasOperation_t op, int n, double ** L, double ** B, int batch)
    {
        const double one = 1;
        return cublasDtrsmBatched(h, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, op, CUBLAS_DIAG_NON_UNIT, n, 1,
                                  &one, L, n, B, n, batch);
    }

    cublasStatus_t trsvBatched(cublasHandle_t h, cublasOperation_t op, int n, float ** L, float ** B, int batch)
    {
        const float one = 1;
        return cublasStrsmBatched(h, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, op, CUBLAS_DIAG_NON_UNIT, n, 1,
                                  &one, L, n, B, n, batch);
    }

    // the special functions of each precision
    __device__ inline double erfcx_(double x) { return erfcx(x); }
    __device__ inline float erfcx_(float x) { return erfcxf(x); }
    __device__ inline double normcdfinv_(double x) { return normcdfinv(x); }
    __device__ inline float normcdfinv_(float x) { return normcdfinvf(x); }

    // log Phi(x), as scipy's log_ndtr
    template <typename T>
    __device__ T log_ndtr(T x)
    {
        const T half = T(0.5), rsqrt2 = T(M_SQRT1_2);
        if (x < T(-1)) {
            T t = -x*rsqrt2;
            return log(half*erfcx_(t)) - half*x*x;
        }
        return log1p(-half*erfc(x*rsqrt2));
    }

    // Phi^-1(exp(q)), q <= 0, as scipy's ndtri_exp
    template <typename T>
    __device__ T ndtri_exp(T q)
    {
        // where exp(q) stays normal
        const T floor = sizeof(T) == 4 ? T(-80) : T(-700);
        if (q > T(-0.6931471805599453)) return -normcdfinv_(-expm1(q));
        if (q > floor) return normcdfinv_(exp(q));
        // the far tail: newton on log Phi from the leading term
        T x = -sqrt(T(-2)*q);
        for (int it = 0; it < 6; ++it) {
            T lp = log_ndtr(x);
            x -= (lp - q)/exp(T(-0.5)*x*x - lp - T(0.9189385332046727));
        }
        return x;
    }

    // the prior scale of sample {s}
    template <typename io_t, typename T>
    __device__ T scale(const io_t * theta, size_t parameters, size_t s, long column, double sigma)
    {
        return column < 0 ? T(sigma) : pow(T(10), T(theta[s*parameters + column]));
    }

    // F[s][p][k], the cumulative slip fraction of patch p in bin k, the mean over its source points
    // of the average over the bin ("bin", mode 2) or the value at its end ("step", mode 1)
    template <typename io_t, typename T>
    __global__ void fractions(const io_t * theta, size_t parameters, const io_t * TI, size_t stride,
        size_t start, size_t batch, int Np, int Nsp, int bins, T dt, int kind, T peak, int mode,
        int fronts, const long * originColumns, T origin, const long * riseColumns, T * F)
    {
        size_t id = threadIdx.x + size_t(blockIdx.x)*blockDim.x;
        if (id >= batch*Np) return;
        size_t s = id / Np;
        int p = id % Np;
        size_t row = start + s;
        const io_t * M = theta + row*parameters;
        T Tr = M[riseColumns[p]];
        T * f = F + id*bins;
        for (int k = 0; k < bins; ++k) f[k] = 0;
        for (int q = 0; q < Nsp; ++q) {
            // the earliest arrival over the fronts
            T t = T(INFINITY);
            for (int c = 0; c < fronts; ++c) {
                T o = originColumns[c] < 0 ? origin : T(M[originColumns[c]]);
                t = fmin(t, T(TI[c*stride + row*size_t(Np*Nsp) + p*Nsp + q]) + o);
            }
            T V0, V0_Tr, W0, W0_Tr;
            altar::models::seismic::stf_cumulative<T>(-t, Tr, kind, peak, V0, V0_Tr, W0, W0_Tr);
            for (int k = 0; k < bins; ++k) {
                T V1, V1_Tr, W1, W1_Tr;
                altar::models::seismic::stf_cumulative<T>(dt*(k + 1) - t, Tr, kind, peak, V1, V1_Tr, W1, W1_Tr);
                f[k] += (mode == 1 ? V1 : (W1 - W0)/dt)/Nsp;
                W0 = W1;
            }
        }
    }

    // P[s] = C0^-1 / sigma^2 + Ps + sum_c GG_c o FRF_c, in the order of the bounded slips last
    template <typename io_t, typename T>
    __global__ void assemble(const io_t * theta, size_t parameters, size_t start, size_t batch, int n,
        int Np, int C, const int * order, const T * C0inv, const T * Ps, const T * GG,
        const T * FRF, long scaleColumn, double sigma, T * P)
    {
        size_t id = threadIdx.x + size_t(blockIdx.x)*blockDim.x;
        size_t nn = size_t(n)*n;
        if (id >= batch*nn) return;
        size_t s = id / nn;
        int i = (id % nn) / n, j = id % n;
        int oi = order[i], oj = order[j];
        T sd = scale<io_t, T>(theta, parameters, start + s, scaleColumn, sigma);
        T v = C0inv[size_t(oi)*n + oj]/(sd*sd) + Ps[size_t(oi)*n + oj];
        for (int c = 0; c < C; ++c) {
            v += GG[c*nn + size_t(oi)*n + oj]*FRF[(c*batch + s)*size_t(Np)*Np + (oi % Np)*Np + (oj % Np)];
        }
        P[id] = v;
    }

    // b[s][i] = bs[i] + sum_c sum_k F[s][p(i)][k] GSD_c[i][k], in the order of the bounded slips last
    template <typename T>
    __global__ void rhs(size_t batch, int n, int Np, int C, int bins, const int * order, const T * F,
        const T * GSD, const T * bs, T * b)
    {
        size_t id = threadIdx.x + size_t(blockIdx.x)*blockDim.x;
        if (id >= batch*n) return;
        size_t s = id / n;
        int oi = order[id % n];
        const T * f = F + (s*Np + oi % Np)*bins;
        T v = bs[oi];
        for (int c = 0; c < C; ++c) {
            const T * g = GSD + (size_t(c)*n + oi)*bins;
            for (int k = 0; k < bins; ++k) v += f[k]*g[k];
        }
        b[id] = v;
    }

    // lnl[s] = constant + |z|^2/2 - log|L| - n log sigma, with z = L^-1 b in {z}, summed in double
    template <typename io_t, typename T>
    __global__ void finalize(const io_t * theta, size_t parameters, size_t start, size_t batch, int n,
        const T * L, const T * z, const int * info, long scaleColumn, double sigma,
        double constant, double * lnl)
    {
        size_t s = threadIdx.x + size_t(blockIdx.x)*blockDim.x;
        if (s >= batch) return;
        if (info[s] != 0) { lnl[s] = -INFINITY; return; }
        const T * l = L + s*size_t(n)*n;
        const T * zs = z + s*n;
        double quad = 0, logdet = 0;
        for (int i = 0; i < n; ++i) {
            double zi = zs[i];
            quad += zi*zi;
            logdet += log(double(l[size_t(i)*n + i]));
        }
        double sd = scale<io_t, double>(theta, parameters, start + s, scaleColumn, sigma);
        lnl[s] = constant + 0.5*quad - logdet - n*log(sd);
    }

    // lnl[s] += log Pr(the last k slips >= 0) by GHK, one block a sample, threads over the
    // paths; L lower, column major, L(r, c) at L[c n + r]; y the scratch of the paths, (k, paths);
    // the weights averaged in double
    template <typename T>
    __global__ void ghk(int n, int k, const T * L, const T * mean, const T * uniforms, int paths,
        T * y, double * lnl)
    {
        size_t s = blockIdx.x;
        const T * l = L + s*size_t(n)*n;
        const T * mu = mean + s*n;
        T * ys = y + s*size_t(k)*paths;
        double top = -INFINITY, sum = 0;
        for (int t = threadIdx.x; t < paths; t += blockDim.x) {
            double logw = 0;
            for (int i = k - 1; i >= 0; --i) {
                int g = n - k + i;
                T acc = 0;
                for (int j = i + 1; j < k; ++j) acc += ys[size_t(j)*paths + t]*l[size_t(g)*n + n - k + j];
                T Lgg = l[size_t(g)*n + g];
                T lp = log_ndtr(Lgg*mu[g] - acc);
                logw += lp;
                T z = -ndtri_exp(T(log1p(-uniforms[size_t(t)*k + i])) + lp);
                ys[size_t(i)*paths + t] = (z - acc)/Lgg;
            }
            if (logw > top) { sum = sum*exp(top - logw) + 1; top = logw; }
            else sum += exp(logw - top);
        }
        // the log of the mean of the weights over the block
        __shared__ double tops[128], sums[128];
        tops[threadIdx.x] = top;
        sums[threadIdx.x] = sum;
        __syncthreads();
        if (threadIdx.x == 0) {
            double M = -INFINITY;
            for (int t = 0; t < blockDim.x; ++t) M = fmax(M, tops[t]);
            double total = 0;
            for (int t = 0; t < blockDim.x; ++t) if (sums[t] > 0) total += sums[t]*exp(tops[t] - M);
            lnl[s] += M + log(total) - log(double(paths));
        }
    }

    // out[start + s] = lnl[s]
    template <typename io_t>
    __global__ void store(size_t start, size_t batch, const double * lnl, io_t * out)
    {
        size_t s = threadIdx.x + size_t(blockIdx.x)*blockDim.x;
        if (s < batch) out[start + s] = io_t(lnl[s]);
    }

    // a device copy of {count} values
    template <typename T>
    T * upload(const T * host, size_t count)
    {
        T * device = nullptr;
        cudaSafeCall(cudaMalloc(&device, count*sizeof(T)));
        cudaSafeCall(cudaMemcpy(device, host, count*sizeof(T), cudaMemcpyHostToDevice));
        return device;
    }

    // a device copy of {count} doubles as {T}
    template <typename T>
    T * upload_as(const double * host, size_t count)
    {
        std::vector<T> cast(host, host + count);
        return upload(cast.data(), count);
    }
}


// constructor
template <typename io_t, typename real_t>
altar::models::seismic::cudaGaussianSlip<io_t, real_t>::
cudaGaussianSlip(size_t Np, size_t Nsp, size_t bins, size_t components, double dt, int kind,
                 double peak, int mode, size_t fronts, const long * originColumns, double origin,
                 const long * riseColumns, long scaleColumn, double sigma, const double * GG,
                 const double * GSD, const double * Rinv, const double * C0inv, const double * Ps,
                 const double * bs, const int * order, size_t bounded, const double * uniforms,
                 size_t paths, double constant, size_t chunk) :
    _Np(Np), _Nsp(Nsp), _bins(bins), _C(components), _n(2*Np), _fronts(fronts), _bounded(bounded),
    _paths(paths), _chunk_size(chunk), _dt(dt), _peak(peak), _origin(origin), _sigma(sigma),
    _constant(constant), _kind(kind), _mode(mode), _scale(scaleColumn)
{
    size_t n = _n, C = _C;
    _originColumns = upload(originColumns, fronts);
    _riseColumns = upload(riseColumns, Np);
    _order = upload(order, n);
    _GG = upload_as<real_t>(GG, C*n*n);
    _GSD = upload_as<real_t>(GSD, C*n*bins);
    _Rinv = upload_as<real_t>(Rinv, C*bins*bins);
    _C0inv = upload_as<real_t>(C0inv, n*n);
    _Ps = upload_as<real_t>(Ps, n*n);
    _bs = upload_as<real_t>(bs, n);
    if (bounded) {
        _uniforms = upload_as<real_t>(uniforms, paths*bounded);
        cudaSafeCall(cudaMalloc(&_y, chunk*bounded*paths*sizeof(real_t)));
    }
    cudaSafeCall(cudaMalloc(&_F, chunk*Np*bins*sizeof(real_t)));
    cudaSafeCall(cudaMalloc(&_FR, chunk*Np*bins*sizeof(real_t)));
    cudaSafeCall(cudaMalloc(&_FRF, C*chunk*Np*Np*sizeof(real_t)));
    cudaSafeCall(cudaMalloc(&_P, chunk*n*n*sizeof(real_t)));
    cudaSafeCall(cudaMalloc(&_b, chunk*n*sizeof(real_t)));
    cudaSafeCall(cudaMalloc(&_mean, chunk*n*sizeof(real_t)));
    cudaSafeCall(cudaMalloc(&_lnl, chunk*sizeof(double)));
    cudaSafeCall(cudaMalloc(&_info, chunk*sizeof(int)));
    // the pointers of the batched factorization and solves
    std::vector<real_t *> Pp(chunk), bp(chunk), mp(chunk);
    for (size_t s = 0; s < chunk; ++s) { Pp[s] = _P + s*n*n; bp[s] = _b + s*n; mp[s] = _mean + s*n; }
    _Parray = upload(Pp.data(), chunk);
    _barray = upload(bp.data(), chunk);
    _marray = upload(mp.data(), chunk);
    cublasSafeCall(cublasCreate(&_cublas));
    cusolverSafeCall(cusolverDnCreate(&_cusolver), "cusolverDnCreate");
}


// destructor
template <typename io_t, typename real_t>
altar::models::seismic::cudaGaussianSlip<io_t, real_t>::
~cudaGaussianSlip()
{
    for (void * p : {(void *)_originColumns, (void *)_riseColumns, (void *)_order, (void *)_GG,
                     (void *)_GSD, (void *)_Rinv, (void *)_C0inv, (void *)_Ps, (void *)_bs, (void *)_uniforms, (void *)_F,
                     (void *)_FR, (void *)_FRF, (void *)_P, (void *)_b, (void *)_mean, (void *)_y,
                     (void *)_Parray, (void *)_barray, (void *)_marray, (void *)_lnl, (void *)_info}) {
        if (p) cudaFree(p);
    }
    cublasDestroy(_cublas);
    cusolverDnDestroy(_cusolver);
}


// the cumulative slip fractions of {batch} samples from row {start}
template <typename io_t, typename real_t>
void
altar::models::seismic::cudaGaussianSlip<io_t, real_t>::
fractions(const io_t * theta, size_t parameters, const io_t * TI, size_t stride, real_t * F,
          size_t batch, size_t start, cudaStream_t stream) const
{
    size_t threads = batch*_Np;
    ::fractions<io_t, real_t><<<IDIVUP(threads, BLOCKDIM), BLOCKDIM, 0, stream>>>
        (theta, parameters, TI, stride, start, batch, int(_Np), int(_Nsp), int(_bins), real_t(_dt), _kind,
         real_t(_peak), _mode, int(_fronts), _originColumns, real_t(_origin), _riseColumns, F);
    cudaCheckError("cudaGaussianSlip.fractions");
}


// the marginal log likelihoods, a chunk of samples at a time
template <typename io_t, typename real_t>
void
altar::models::seismic::cudaGaussianSlip<io_t, real_t>::
loglikelihood(const io_t * theta, size_t parameters, const io_t * TI, size_t stride, io_t * out,
              size_t batch)
{
    for (size_t start = 0; start < batch; start += _chunk_size) {
        _chunk(theta, parameters, TI, stride, out, start, std::min(_chunk_size, batch - start));
    }
}


template <typename io_t, typename real_t>
void
altar::models::seismic::cudaGaussianSlip<io_t, real_t>::
_chunk(const io_t * theta, size_t parameters, const io_t * TI, size_t stride, io_t * out,
       size_t start, size_t batch)
{
    int n = _n, Np = _Np, nt = _bins;
    fractions(theta, parameters, TI, stride, _F, batch, start);
    // per component, F R^-1 then (F R^-1) F^T; column major, a row-major (Np, nt) F is F^T
    const real_t one = 1, zero = 0;
    for (size_t c = 0; c < _C; ++c) {
        cublasSafeCall(gemmStridedBatched(_cublas, CUBLAS_OP_N, CUBLAS_OP_N, nt, Np, nt, &one,
            _Rinv + c*nt*nt, nt, 0, _F, nt, size_t(Np)*nt, &zero, _FR, nt, size_t(Np)*nt, batch));
        cublasSafeCall(gemmStridedBatched(_cublas, CUBLAS_OP_T, CUBLAS_OP_N, Np, Np, nt, &one,
            _FR, nt, size_t(Np)*nt, _F, nt, size_t(Np)*nt, &zero, _FRF + c*batch*Np*Np, Np, size_t(Np)*Np,
            batch));
    }
    size_t entries = batch*size_t(n)*n;
    ::assemble<io_t, real_t><<<IDIVUP(entries, BLOCKDIM), BLOCKDIM>>>
        (theta, parameters, start, batch, n, Np, int(_C), _order, _C0inv, _Ps, _GG, _FRF, _scale, _sigma, _P);
    cudaCheckError("cudaGaussianSlip.assemble");
    ::rhs<real_t><<<IDIVUP(batch*n, BLOCKDIM), BLOCKDIM>>>(batch, n, Np, int(_C), nt, _order, _F, _GSD, _bs, _b);
    cudaCheckError("cudaGaussianSlip.rhs");
    // P = L L^T; z = L^-1 b in {b}; the mean L^-T z
    cusolverSafeCall(potrfBatched(_cusolver, n, _Parray, _info, int(batch)), "potrfBatched");
    cublasSafeCall(trsvBatched(_cublas, CUBLAS_OP_N, n, _Parray, _barray, int(batch)));
    ::finalize<io_t, real_t><<<IDIVUP(batch, BLOCKDIM), BLOCKDIM>>>
        (theta, parameters, start, batch, n, _P, _b, _info, _scale, _sigma, _constant, _lnl);
    cudaCheckError("cudaGaussianSlip.finalize");
    if (_bounded) {
        cudaSafeCall(cudaMemcpyAsync(_mean, _b, batch*n*sizeof(real_t), cudaMemcpyDeviceToDevice));
        cublasSafeCall(trsvBatched(_cublas, CUBLAS_OP_T, n, _Parray, _marray, int(batch)));
        ::ghk<real_t><<<batch, 128>>>(n, int(_bounded), _P, _mean, _uniforms, int(_paths), _y, _lnl);
        cudaCheckError("cudaGaussianSlip.ghk");
    }
    ::store<io_t><<<IDIVUP(batch, BLOCKDIM), BLOCKDIM>>>(start, batch, _lnl, out);
    cudaCheckError("cudaGaussianSlip.store");
}


// the precisions: all double; float theta and likelihoods, double algebra; all float
template class altar::models::seismic::cudaGaussianSlip<double, double>;
template class altar::models::seismic::cudaGaussianSlip<float, double>;
template class altar::models::seismic::cudaGaussianSlip<float, float>;

// end of file
