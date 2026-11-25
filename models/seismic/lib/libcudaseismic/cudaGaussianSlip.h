// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//

// the marginal likelihood of the rupture fronts of kinematicgaussianslip, the slips integrated out
// under their gaussian prior, on the gpu: the cumulative slip fractions F from the arrival times at
// the source points, then per component F R^-1 F^T, the system
// P = C0^-1 / sigma^2 + Ps + sum_c GG_c o (F R_c^-1 F^T) and b, its batched cholesky factor, and,
// with bounded slips, their box probability by GHK; see seismic/marginal.py, its cpu twin.
// {io_t} is the type of theta, the arrival times and the likelihoods; {real_t} that of the
// fractions and the linear algebra; the sums of the likelihoods are in double

#if !defined(altar_models_seismic_cudaGaussianSlip_h)
#define altar_models_seismic_cudaGaussianSlip_h

#include <cstddef>
#include <cublas_v2.h>
#include <cusolverDn.h>

namespace altar::models::seismic {
    template <typename io_t, typename real_t> class cudaGaussianSlip;
}


template <typename io_t, typename real_t>
class altar::models::seismic::cudaGaussianSlip
{
public:
    /// out[s] <- the marginal log likelihood of row s of {theta}, (batch, parameters); {TI} holds
    /// the arrival times at the source points of each nucleation point, front f of sample s at
    /// TI + f*stride + s*points
    void loglikelihood(const io_t * theta, size_t parameters, const io_t * TI, size_t stride,
                       io_t * out, size_t batch);
    /// F <- the cumulative slip fractions, (batch, patches, bins)
    void fractions(const io_t * theta, size_t parameters, const io_t * TI, size_t stride,
                   real_t * F, size_t batch, size_t start = 0, cudaStream_t stream = 0) const;

public:
    // the sizes; the source time function; the columns of theta: the origin of each front, or
    // none and {origin}, the rise time of each patch, the slip scale, or none and {sigma}; the
    // system, with Ps and bs the constant parts of P and b, from static data, and {order}, the
    // original index of each slip in the order of the bounded slips last; the sobol points of GHK, (paths, bounded); the constant of the likelihood; the samples at once
    cudaGaussianSlip(size_t Np, size_t Nsp, size_t bins, size_t components, double dt, int kind,
                     double peak, int mode, size_t fronts, const long * originColumns, double origin,
                     const long * riseColumns, long scaleColumn, double sigma, const double * GG,
                     const double * GSD, const double * Rinv, const double * C0inv, const double * Ps,
                     const double * bs, const int * order, size_t bounded, const double * uniforms,
                     size_t paths, double constant, size_t chunk);
    ~cudaGaussianSlip();

private:
    void _chunk(const io_t * theta, size_t parameters, const io_t * TI, size_t stride,
                io_t * out, size_t start, size_t batch);

    size_t _Np, _Nsp, _bins, _C, _n, _fronts, _bounded, _paths, _chunk_size;
    double _dt, _peak, _origin, _sigma, _constant;
    int _kind, _mode;
    long _scale;
    // on the device
    long * _originColumns = nullptr, * _riseColumns = nullptr;
    int * _order = nullptr;
    real_t * _GG = nullptr, * _GSD = nullptr, * _Rinv = nullptr, * _C0inv = nullptr, * _uniforms = nullptr;
    real_t * _Ps = nullptr, * _bs = nullptr;
    // work, for a chunk of samples
    real_t * _F = nullptr, * _FR = nullptr, * _FRF = nullptr, * _P = nullptr, * _b = nullptr;
    real_t * _mean = nullptr, * _y = nullptr;
    real_t ** _Parray = nullptr, ** _barray = nullptr, ** _marray = nullptr;
    double * _lnl = nullptr;
    int * _info = nullptr;
    cublasHandle_t _cublas;
    cusolverDnHandle_t _cusolver;

    cudaGaussianSlip(const cudaGaussianSlip &) = delete;
    const cudaGaussianSlip & operator=(const cudaGaussianSlip &) = delete;
};

#endif

// end of file
