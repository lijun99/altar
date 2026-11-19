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

    // log Phi(x), as scipy's log_ndtr
    __device__ double log_ndtr(double x)
    {
        if (x < -1.0) {
            double t = -x*M_SQRT1_2;
            return log(0.5*erfcx(t)) - 0.5*x*x;
        }
        return log1p(-0.5*erfc(x*M_SQRT1_2));
    }

    // Phi^-1(exp(q)), q <= 0, as scipy's ndtri_exp
    __device__ double ndtri_exp(double q)
    {
        if (q > -0.6931471805599453) return -normcdfinv(-expm1(q));
        if (q > -700.0) return normcdfinv(exp(q));
        // the far tail: newton on log Phi from the leading term
        double x = -sqrt(-2*q);
        for (int it = 0; it < 6; ++it) {
            double lp = log_ndtr(x);
            x -= (lp - q)/exp(-0.5*x*x - lp - 0.9189385332046727);
        }
        return x;
    }

    // the prior scale of sample {s}
    __device__ double scale(const double * theta, size_t parameters, size_t s, long column, double sigma)
    {
        return column < 0 ? sigma : pow(10.0, theta[s*parameters + column]);
    }

    // F[s][p][k], the cumulative slip fraction of patch p in bin k, the mean over its source points
    // of the average over the bin ("bin", mode 2) or the value at its end ("step", mode 1)
    __global__ void fractions(const double * theta, size_t parameters, const double * TI, size_t stride,
        size_t start, size_t batch, int Np, int Nsp, int bins, double dt, int kind, double peak, int mode,
        int fronts, const long * originColumns, double origin, const long * riseColumns, double * F)
    {
        size_t id = threadIdx.x + size_t(blockIdx.x)*blockDim.x;
        if (id >= batch*Np) return;
        size_t s = id / Np;
        int p = id % Np;
        size_t row = start + s;
        const double * M = theta + row*parameters;
        double Tr = M[riseColumns[p]];
        double * f = F + id*bins;
        for (int k = 0; k < bins; ++k) f[k] = 0;
        for (int q = 0; q < Nsp; ++q) {
            // the earliest arrival over the fronts
            double T = 1e300;
            for (int c = 0; c < fronts; ++c) {
                double o = originColumns[c] < 0 ? origin : M[originColumns[c]];
                T = fmin(T, TI[c*stride + row*size_t(Np*Nsp) + p*Nsp + q] + o);
            }
            double V0, V0_Tr, W0, W0_Tr;
            altar::models::seismic::stf_cumulative(-T, Tr, kind, peak, V0, V0_Tr, W0, W0_Tr);
            for (int k = 0; k < bins; ++k) {
                double V1, V1_Tr, W1, W1_Tr;
                altar::models::seismic::stf_cumulative(dt*(k + 1) - T, Tr, kind, peak, V1, V1_Tr, W1, W1_Tr);
                f[k] += (mode == 1 ? V1 : (W1 - W0)/dt)/Nsp;
                W0 = W1;
            }
        }
    }

    // P[s] = C0^-1 / sigma^2 + sum_c GG_c o FRF_c, in the order of the bounded slips last
    __global__ void assemble(const double * theta, size_t parameters, size_t start, size_t batch, int n,
        int Np, int C, const int * order, const double * C0inv, const double * GG, const double * FRF,
        long scaleColumn, double sigma, double * P)
    {
        size_t id = threadIdx.x + size_t(blockIdx.x)*blockDim.x;
        size_t nn = size_t(n)*n;
        if (id >= batch*nn) return;
        size_t s = id / nn;
        int i = (id % nn) / n, j = id % n;
        int oi = order[i], oj = order[j];
        double sd = scale(theta, parameters, start + s, scaleColumn, sigma);
        double v = C0inv[size_t(oi)*n + oj]/(sd*sd);
        for (int c = 0; c < C; ++c) {
            v += GG[c*nn + size_t(oi)*n + oj]*FRF[(c*batch + s)*size_t(Np)*Np + (oi % Np)*Np + (oj % Np)];
        }
        P[id] = v;
    }

    // b[s][i] = sum_c sum_k F[s][p(i)][k] GSD_c[i][k], in the order of the bounded slips last
    __global__ void rhs(size_t batch, int n, int Np, int C, int bins, const int * order, const double * F,
        const double * GSD, double * b)
    {
        size_t id = threadIdx.x + size_t(blockIdx.x)*blockDim.x;
        if (id >= batch*n) return;
        size_t s = id / n;
        int oi = order[id % n];
        const double * f = F + (s*Np + oi % Np)*bins;
        double v = 0;
        for (int c = 0; c < C; ++c) {
            const double * g = GSD + (size_t(c)*n + oi)*bins;
            for (int k = 0; k < bins; ++k) v += f[k]*g[k];
        }
        b[id] = v;
    }

    // out[start + s] = constant + |z|^2/2 - log|L| - n log sigma, with z = L^-1 b in {z}
    __global__ void finalize(const double * theta, size_t parameters, size_t start, size_t batch, int n,
        const double * L, const double * z, const int * info, long scaleColumn, double sigma,
        double constant, double * out)
    {
        size_t s = threadIdx.x + size_t(blockIdx.x)*blockDim.x;
        if (s >= batch) return;
        if (info[s] != 0) { out[start + s] = -INFINITY; return; }
        const double * l = L + s*size_t(n)*n;
        const double * zs = z + s*n;
        double quad = 0, logdet = 0;
        for (int i = 0; i < n; ++i) {
            quad += zs[i]*zs[i];
            logdet += log(l[size_t(i)*n + i]);
        }
        double sd = scale(theta, parameters, start + s, scaleColumn, sigma);
        out[start + s] = constant + 0.5*quad - logdet - n*log(sd);
    }

    // out[start + s] += log Pr(the last k slips >= 0) by GHK, one block a sample, threads over the
    // paths; L lower, column major, L(r, c) at L[c n + r]; y the scratch of the paths, (k, paths)
    __global__ void ghk(size_t start, int n, int k, const double * L, const double * mean,
        const double * uniforms, int paths, double * y, double * out)
    {
        size_t s = blockIdx.x;
        const double * l = L + s*size_t(n)*n;
        const double * mu = mean + s*n;
        double * ys = y + s*size_t(k)*paths;
        double top = -INFINITY, sum = 0;
        for (int t = threadIdx.x; t < paths; t += blockDim.x) {
            double logw = 0;
            for (int i = k - 1; i >= 0; --i) {
                int g = n - k + i;
                double acc = 0;
                for (int j = i + 1; j < k; ++j) acc += ys[size_t(j)*paths + t]*l[size_t(g)*n + n - k + j];
                double Lgg = l[size_t(g)*n + g];
                double lp = log_ndtr(Lgg*mu[g] - acc);
                logw += lp;
                double z = -ndtri_exp(log1p(-uniforms[size_t(t)*k + i]) + lp);
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
            out[start + s] += M + log(total) - log(double(paths));
        }
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
}


// constructor
altar::models::seismic::cudaGaussianSlip::
cudaGaussianSlip(size_t Np, size_t Nsp, size_t bins, size_t components, double dt, int kind,
                 double peak, int mode, size_t fronts, const long * originColumns, double origin,
                 const long * riseColumns, long scaleColumn, double sigma, const double * GG,
                 const double * GSD, const double * Rinv, const double * C0inv, const int * order,
                 size_t bounded, const double * uniforms, size_t paths, double constant, size_t chunk) :
    _Np(Np), _Nsp(Nsp), _bins(bins), _C(components), _n(2*Np), _fronts(fronts), _bounded(bounded),
    _paths(paths), _chunk_size(chunk), _dt(dt), _peak(peak), _origin(origin), _sigma(sigma),
    _constant(constant), _kind(kind), _mode(mode), _scale(scaleColumn)
{
    size_t n = _n, C = _C;
    _originColumns = upload(originColumns, fronts);
    _riseColumns = upload(riseColumns, Np);
    _order = upload(order, n);
    _GG = upload(GG, C*n*n);
    _GSD = upload(GSD, C*n*bins);
    _Rinv = upload(Rinv, C*bins*bins);
    _C0inv = upload(C0inv, n*n);
    if (bounded) {
        _uniforms = upload(uniforms, paths*bounded);
        cudaSafeCall(cudaMalloc(&_y, chunk*bounded*paths*sizeof(double)));
    }
    cudaSafeCall(cudaMalloc(&_F, chunk*Np*bins*sizeof(double)));
    cudaSafeCall(cudaMalloc(&_FR, chunk*Np*bins*sizeof(double)));
    cudaSafeCall(cudaMalloc(&_FRF, C*chunk*Np*Np*sizeof(double)));
    cudaSafeCall(cudaMalloc(&_P, chunk*n*n*sizeof(double)));
    cudaSafeCall(cudaMalloc(&_b, chunk*n*sizeof(double)));
    cudaSafeCall(cudaMalloc(&_mean, chunk*n*sizeof(double)));
    cudaSafeCall(cudaMalloc(&_info, chunk*sizeof(int)));
    // the pointers of the batched factorization and solves
    std::vector<double *> Ps(chunk), bs(chunk), ms(chunk);
    for (size_t s = 0; s < chunk; ++s) { Ps[s] = _P + s*n*n; bs[s] = _b + s*n; ms[s] = _mean + s*n; }
    _Parray = upload(Ps.data(), chunk);
    _barray = upload(bs.data(), chunk);
    _marray = upload(ms.data(), chunk);
    cublasSafeCall(cublasCreate(&_cublas));
    cusolverSafeCall(cusolverDnCreate(&_cusolver), "cusolverDnCreate");
}


// destructor
altar::models::seismic::cudaGaussianSlip::
~cudaGaussianSlip()
{
    for (void * p : {(void *)_originColumns, (void *)_riseColumns, (void *)_order, (void *)_GG,
                     (void *)_GSD, (void *)_Rinv, (void *)_C0inv, (void *)_uniforms, (void *)_F,
                     (void *)_FR, (void *)_FRF, (void *)_P, (void *)_b, (void *)_mean, (void *)_y,
                     (void *)_Parray, (void *)_barray, (void *)_marray, (void *)_info}) {
        if (p) cudaFree(p);
    }
    cublasDestroy(_cublas);
    cusolverDnDestroy(_cusolver);
}


// the cumulative slip fractions of {batch} samples from row {start}
void
altar::models::seismic::cudaGaussianSlip::
fractions(const double * theta, size_t parameters, const double * TI, size_t stride, double * F,
          size_t batch, size_t start, cudaStream_t stream) const
{
    size_t threads = batch*_Np;
    ::fractions<<<IDIVUP(threads, BLOCKDIM), BLOCKDIM, 0, stream>>>
        (theta, parameters, TI, stride, start, batch, int(_Np), int(_Nsp), int(_bins), _dt, _kind,
         _peak, _mode, int(_fronts), _originColumns, _origin, _riseColumns, F);
    cudaCheckError("cudaGaussianSlip.fractions");
}


// the marginal log likelihoods, a chunk of samples at a time
void
altar::models::seismic::cudaGaussianSlip::
loglikelihood(const double * theta, size_t parameters, const double * TI, size_t stride, double * out,
              size_t batch)
{
    for (size_t start = 0; start < batch; start += _chunk_size) {
        _chunk(theta, parameters, TI, stride, out, start, std::min(_chunk_size, batch - start));
    }
}


void
altar::models::seismic::cudaGaussianSlip::
_chunk(const double * theta, size_t parameters, const double * TI, size_t stride, double * out,
       size_t start, size_t batch)
{
    int n = _n, Np = _Np, nt = _bins;
    fractions(theta, parameters, TI, stride, _F, batch, start);
    // per component, F R^-1 then (F R^-1) F^T; column major, a row-major (Np, nt) F is F^T
    const double one = 1, zero = 0;
    for (size_t c = 0; c < _C; ++c) {
        cublasSafeCall(cublasDgemmStridedBatched(_cublas, CUBLAS_OP_N, CUBLAS_OP_N, nt, Np, nt, &one,
            _Rinv + c*nt*nt, nt, 0, _F, nt, size_t(Np)*nt, &zero, _FR, nt, size_t(Np)*nt, batch));
        cublasSafeCall(cublasDgemmStridedBatched(_cublas, CUBLAS_OP_T, CUBLAS_OP_N, Np, Np, nt, &one,
            _FR, nt, size_t(Np)*nt, _F, nt, size_t(Np)*nt, &zero, _FRF + c*batch*Np*Np, Np, size_t(Np)*Np,
            batch));
    }
    size_t entries = batch*size_t(n)*n;
    ::assemble<<<IDIVUP(entries, BLOCKDIM), BLOCKDIM>>>
        (theta, parameters, start, batch, n, Np, int(_C), _order, _C0inv, _GG, _FRF, _scale, _sigma, _P);
    cudaCheckError("cudaGaussianSlip.assemble");
    ::rhs<<<IDIVUP(batch*n, BLOCKDIM), BLOCKDIM>>>(batch, n, Np, int(_C), nt, _order, _F, _GSD, _b);
    cudaCheckError("cudaGaussianSlip.rhs");
    // P = L L^T; z = L^-1 b in {b}; the mean L^-T z
    cusolverSafeCall(cusolverDnDpotrfBatched(_cusolver, CUBLAS_FILL_MODE_LOWER, n, _Parray, n, _info,
                                             int(batch)), "cusolverDnDpotrfBatched");
    cublasSafeCall(cublasDtrsmBatched(_cublas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
        CUBLAS_DIAG_NON_UNIT, n, 1, &one, _Parray, n, _barray, n, int(batch)));
    ::finalize<<<IDIVUP(batch, BLOCKDIM), BLOCKDIM>>>
        (theta, parameters, start, batch, n, _P, _b, _info, _scale, _sigma, _constant, out);
    cudaCheckError("cudaGaussianSlip.finalize");
    if (!_bounded) return;
    cudaSafeCall(cudaMemcpyAsync(_mean, _b, batch*n*sizeof(double), cudaMemcpyDeviceToDevice));
    cublasSafeCall(cublasDtrsmBatched(_cublas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
        CUBLAS_DIAG_NON_UNIT, n, 1, &one, _Parray, n, _marray, n, int(batch)));
    ::ghk<<<batch, 128>>>(start, n, int(_bounded), _P, _mean, _uniforms, int(_paths), _y, out);
    cudaCheckError("cudaGaussianSlip.ghk");
}


// end of file
