// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//

// the kinematic model of rectangular patches on the cpu, line for line the arithmetic of
// cudaKinematic_kernels.cu; the sweeps run row by row, which updates the cells in the order of the
// cuda anti-diagonals, since the cells of an anti-diagonal don't neighbor each other

#include "Kinematic.h"

#include <algorithm>
#include <cmath>

namespace {
    // the share of the gradient {g} of min(a, b) that goes to {a}: all of it, half on a tie, or none
    template <typename TYPE>
    TYPE dmin(TYPE a, TYPE b, TYPE g)
    {
        return a < b ? g : (a == b ? g/2 : TYPE(0));
    }

    // the order in which sweep {s} updates the cell {i}, {j}: a cell reads the neighbors it
    // precedes as they were before the sweep, and the ones that precede it as they are after it
    inline int order(int s, int i, int j)
    {
        switch (s) {
            case 0: return i + j;
            case 1: return j - i;
            case 2: return -(i + j);
            default: return i - j;
        }
    }

    // the cell of sweep {s} at its step {k}: along strike ascending in sweeps 0 and 3, down dip
    // ascending in sweeps 0 and 1
    inline void cell(int s, int k, int Nasf, int Nddf, int & i, int & j)
    {
        int ii = k / Nddf, jj = k % Nddf;
        i = (s == 0 || s == 3) ? ii : Nasf - 1 - ii;
        j = (s == 0 || s == 1) ? jj : Nddf - 1 - jj;
    }
}


// constructor
template <typename TYPE>
altar::models::seismic::Kinematic<TYPE>::
Kinematic(std::size_t Nas, std::size_t Ndd, std::size_t Nmesh, double dsp, std::size_t Nt,
          std::size_t Npt, double dt, const double * t0s, const std::size_t * idx_map) :
    _Nas(Nas), _Ndd(Ndd), _Nmesh(Nmesh), _Nt(Nt), _Npt(Npt), _Npatch(Nas*Ndd),
    _Nasf((Nas+2)*Nmesh), _Nddf((Ndd+2)*Nmesh), _NGb(2*Nas*Ndd*Nt),
    _dsp(dsp), _dt(dt), _h(TYPE(dsp)/Nmesh),
    _t0s(t0s, t0s + Nas*Ndd), _idx(idx_map, idx_map + 4*Nas*Ndd + 2)
{}


// Mb for each sample
template <typename TYPE>
void
altar::models::seismic::Kinematic<TYPE>::
calculateBigM(const TYPE * theta, TYPE * Mb, std::size_t parameters, std::size_t batch) const
{
    #pragma omp parallel
    {
        std::vector<TYPE> T(cells()), TI0(_Npatch*_Npt*_Npt);
        #pragma omp for schedule(dynamic)
        for (long sample = 0; sample < long(batch); ++sample) {
            const TYPE * M = theta + sample*parameters;
            _seed(M, T.data());
            _sweep(M, T.data(), nullptr);
            _interpolate(T.data(), TI0.data());
            _cast(M, TI0.data(), Mb + sample*_NGb);
        }
    }
}


// the arrival times for each sample
template <typename TYPE>
void
altar::models::seismic::Kinematic<TYPE>::
arrivals(const TYPE * theta, TYPE * T, std::size_t parameters, std::size_t batch) const
{
    #pragma omp parallel for schedule(dynamic)
    for (long sample = 0; sample < long(batch); ++sample) {
        const TYPE * M = theta + sample*parameters;
        _seed(M, T + sample*cells());
        _sweep(M, T + sample*cells(), nullptr);
    }
}


// the gradient: the forward model again, keeping the arrival times between sweeps, then the adjoints
// of the source time functions, the interpolation, and the sweeps and the seeding
template <typename TYPE>
void
altar::models::seismic::Kinematic<TYPE>::
gradient(const TYPE * theta, const TYPE * dMb, TYPE * grad, std::size_t parameters,
         std::size_t batch) const
{
    #pragma omp parallel
    {
        std::vector<TYPE> snapshots((4*_sweeps + 1)*cells()), T(cells()), dT(cells());
        std::vector<TYPE> TI0(_Npatch*_Npt*_Npt), dTI0(_Npatch*_Npt*_Npt);
        #pragma omp for schedule(dynamic)
        for (long sample = 0; sample < long(batch); ++sample) {
            const TYPE * M = theta + sample*parameters;
            TYPE * g = grad + sample*parameters;
            _seed(M, T.data());
            _sweep(M, T.data(), snapshots.data());
            _interpolate(T.data(), TI0.data());
            _castAdjoint(M, TI0.data(), dMb + sample*_NGb, g, dTI0.data());
            std::fill(dT.begin(), dT.end(), TYPE(0));
            _interpolateAdjoint(dTI0.data(), dT.data());
            _sweepAdjoint(M, snapshots.data(), dT.data(), g);
        }
    }
}


// the seeds: straight ray times from the hypocenter within {radius} cells, rising steeply past it
template <typename TYPE>
void
altar::models::seismic::Kinematic<TYPE>::
_seed(const TYPE * M, TYPE * T) const
{
    const int Nas = _Nas, Ndd = _Ndd, Nmesh = _Nmesh, Npatch = _Npatch;
    // the hypocenter, shifted by the padding
    TYPE hypo_strike = M[_idx[4*Npatch]] + _dsp*1.5;
    TYPE hypo_dip = M[_idx[4*Npatch+1]] + _dsp*1.5;
    for (int i = 0; i < int(_Nasf); ++i) {
        for (int j = 0; j < int(_Nddf); ++j) {
            TYPE distance_strike = (i+0.5)*_h - hypo_strike;
            TYPE distance_dip = (j+0.5)*_h - hypo_dip;
            TYPE distance = std::sqrt(distance_strike*distance_strike + distance_dip*distance_dip);
            TYPE far = std::max(TYPE(0), distance - _radius*_h);
            // the rupture velocity of the patch of the cell, clamped to the fault
            int pi = std::min(std::max(i/Nmesh - 1, 0), Nas - 1);
            int pj = std::min(std::max(j/Nmesh - 1, 0), Ndd - 1);
            TYPE vr = M[_idx[3*Npatch + pi*Ndd + pj]];
            T[i*_Nddf + j] = std::min<TYPE>((distance + _steepness*far*far/_h)/vr, _cap);
        }
    }
}


// the upwind update of the cell {i}, {j}
template <typename TYPE>
TYPE
altar::models::seismic::Kinematic<TYPE>::
_upwind(const TYPE * M, const TYPE * T, int i, int j) const
{
    const int Nasf = _Nasf, Nddf = _Nddf, Nmesh = _Nmesh, Nas = _Nas, Ndd = _Ndd;
    int i1 = std::max(0, i-1), i2 = std::min(Nasf-1, i+1);
    int j1 = std::max(0, j-1), j2 = std::min(Nddf-1, j+1);
    TYPE u_old = T[i*Nddf + j];
    TYPE u_xmin = std::min(T[i1*Nddf + j], T[i2*Nddf + j]);
    TYPE u_ymin = std::min(T[i*Nddf + j1], T[i*Nddf + j2]);
    // the patch of the cell
    int pi = i/Nmesh - 1;
    if (pi == -1) pi = 0;
    if (pi >= Nas) pi = Nas - 1;
    int pj = j/Nmesh - 1;
    if (pj == -1) pj = 0;
    if (pj == Ndd) pj = Ndd - 1;
    TYPE f = 1./M[_idx[3*_Npatch + pi*Ndd + pj]];
    TYPE u_new;
    if (std::fabs(u_xmin - u_ymin) >= f*_h) {
        u_new = std::min(u_xmin, u_ymin) + f*_h;
    } else {
        u_new = (u_xmin + u_ymin + std::sqrt(2.*f*f*_h*_h - (u_xmin-u_ymin)*(u_xmin-u_ymin)))/2.0;
    }
    return std::min(u_new, u_old);
}


// the four sweeps, keeping the arrival times before the first and after each in {snapshots}, if any
template <typename TYPE>
void
altar::models::seismic::Kinematic<TYPE>::
_sweep(const TYPE * M, TYPE * T, TYPE * snapshots) const
{
    const int Nasf = _Nasf, Nddf = _Nddf, n = Nasf*Nddf;
    int snapshot = 0;
    if (snapshots) std::copy(T, T + n, snapshots + n*snapshot++);
    for (int iter = 0; iter < _sweeps; ++iter) {
        for (int s = 0; s < 4; ++s) {
            for (int k = 0; k < n; ++k) {
                int i, j;
                cell(s, k, Nasf, Nddf, i, j);
                T[i*Nddf + j] = _upwind(M, T, i, j);
            }
            if (snapshots) std::copy(T, T + n, snapshots + n*snapshot++);
        }
    }
}


// the arrival times at the source points, bilinear in the cells
template <typename TYPE>
void
altar::models::seismic::Kinematic<TYPE>::
_interpolate(const TYPE * T, TYPE * TI0) const
{
    const int Ndd = _Ndd, Nas = _Nas, Npt = _Npt, Nddf = _Nddf;
    TYPE offset = (TYPE(_Nmesh)-0.5) + 0.5*TYPE(_Nmesh)/TYPE(Npt);
    for (int i = 0; i < Ndd*Npt; ++i) {
        TYPE x = offset + TYPE(i)*TYPE(_Nmesh)/TYPE(Npt);
        TYPE xr = x - TYPE(int(x));
        int idx = int(x);
        for (int j = 0; j < Nas*Npt; ++j) {
            TYPE y = offset + TYPE(j)*TYPE(_Nmesh)/TYPE(Npt);
            TYPE yr = y - TYPE(int(y));
            int idy = int(y);
            TYPE f11 = T[idx + idy*Nddf], f21 = T[idx+1 + idy*Nddf];
            TYPE f12 = T[idx + (idy+1)*Nddf], f22 = T[(idx+1) + (idy+1)*Nddf];
            TI0[i + j*(Ndd*Npt)] = f11*(1.0-xr)*(1.0-yr) + f21*xr*(1.0-yr) + f12*(1.0-xr)*yr + f22*xr*yr;
        }
    }
}


// the slips of each patch over time: triangles of the rise time from the arrival time at each of its
// source points, sampled at the start of each time interval
template <typename TYPE>
void
altar::models::seismic::Kinematic<TYPE>::
_cast(const TYPE * M, const TYPE * TI0, TYPE * Mb) const
{
    const int Npatch = _Npatch, Ndd = _Ndd, Npt = _Npt;
    for (int patch = 0; patch < Npatch; ++patch) {
        TYPE ss = M[_idx[patch]], sd = M[_idx[patch+Npatch]];
        TYPE hTr = M[_idx[2*Npatch+patch]]/2.0;
        int idx = patch % Ndd, idy = patch / Ndd;
        for (int i = 0; i < int(_Nt); ++i) {
            TYPE t0 = _t0s[patch] + _dt*i;
            TYPE c = 0.0;
            for (int idx_ti = idx*Npt; idx_ti < (idx+1)*Npt; ++idx_ti) {
                for (int idy_ti = idy*Npt; idy_ti < (idy+1)*Npt; ++idy_ti) {
                    TYPE TI = TI0[idx_ti + idy_ti*(Ndd*Npt)] + hTr;
                    if (t0 <= (TI-hTr) || t0 >= (TI+hTr)) continue;
                    c += (1.0 - std::abs(TI-t0)/hTr)*(1.0/hTr)/(Npt*Npt);
                }
            }
            Mb[i*(2*Npatch) + patch] = c*ss;
            Mb[i*(2*Npatch) + Npatch + patch] = c*sd;
        }
    }
}


// the adjoint of the source time functions: the gradients with respect to the slips and rise times
// into {grad}, and with respect to the arrival times at the source points into {dTI0}
template <typename TYPE>
void
altar::models::seismic::Kinematic<TYPE>::
_castAdjoint(const TYPE * M, const TYPE * TI0, const TYPE * dMb, TYPE * grad, TYPE * dTI0) const
{
    const int Npatch = _Npatch, Ndd = _Ndd, Npt = _Npt;
    TYPE n2 = Npt*Npt;
    for (int patch = 0; patch < Npatch; ++patch) {
        TYPE ss = M[_idx[patch]], sd = M[_idx[patch+Npatch]];
        TYPE hTr = M[_idx[2*Npatch+patch]]/2.0;
        int idx = patch % Ndd, idy = patch / Ndd;
        for (int idx_ti = idx*Npt; idx_ti < (idx+1)*Npt; ++idx_ti) {
            for (int idy_ti = idy*Npt; idy_ti < (idy+1)*Npt; ++idy_ti) {
                dTI0[idx_ti + idy_ti*(Ndd*Npt)] = 0;
            }
        }
        TYPE gss = 0, gsd = 0, ghTr = 0;
        for (int i = 0; i < int(_Nt); ++i) {
            TYPE t0 = _t0s[patch] + _dt*i;
            TYPE dss = dMb[i*(2*Npatch) + patch], dsd = dMb[i*(2*Npatch) + Npatch + patch];
            TYPE gc = dss*ss + dsd*sd;
            TYPE c = 0;
            for (int idx_ti = idx*Npt; idx_ti < (idx+1)*Npt; ++idx_ti) {
                for (int idy_ti = idy*Npt; idy_ti < (idy+1)*Npt; ++idy_ti) {
                    int k = idx_ti + idy_ti*(Ndd*Npt);
                    TYPE TI = TI0[k] + hTr;
                    if (t0 <= (TI-hTr) || t0 >= (TI+hTr)) continue;
                    TYPE a = TI - t0;
                    TYPE absa = std::abs(a);
                    TYPE sign = TYPE((a > 0) - (a < 0));
                    c += (1.0 - absa/hTr)*(1.0/hTr)/n2;
                    // v = (1 - |a|/h)/h/n2, with a = TI0 + h - t
                    TYPE dv_da = -sign/(hTr*hTr*n2);
                    TYPE dv_dh = (2*absa/hTr - 1)/(hTr*hTr*n2);
                    dTI0[k] += gc*dv_da;
                    ghTr += gc*(dv_dh + dv_da);
                }
            }
            gss += dss*c;
            gsd += dsd*c;
        }
        grad[_idx[patch]] = gss;
        grad[_idx[patch+Npatch]] = gsd;
        grad[_idx[2*Npatch+patch]] = ghTr/2.0;
    }
}


// the adjoint of the interpolation, onto {dT}, which must be zeroed first
template <typename TYPE>
void
altar::models::seismic::Kinematic<TYPE>::
_interpolateAdjoint(const TYPE * dTI0, TYPE * dT) const
{
    const int Ndd = _Ndd, Nas = _Nas, Npt = _Npt, Nddf = _Nddf;
    TYPE offset = (TYPE(_Nmesh)-0.5) + 0.5*TYPE(_Nmesh)/TYPE(Npt);
    for (int i = 0; i < Ndd*Npt; ++i) {
        TYPE x = offset + TYPE(i)*TYPE(_Nmesh)/TYPE(Npt);
        TYPE xr = x - TYPE(int(x));
        int idx = int(x);
        for (int j = 0; j < Nas*Npt; ++j) {
            TYPE y = offset + TYPE(j)*TYPE(_Nmesh)/TYPE(Npt);
            TYPE yr = y - TYPE(int(y));
            int idy = int(y);
            TYPE g = dTI0[i + j*(Ndd*Npt)];
            dT[idx + idy*Nddf] += g*(1.0-xr)*(1.0-yr);
            dT[idx+1 + idy*Nddf] += g*xr*(1.0-yr);
            dT[idx + (idy+1)*Nddf] += g*(1.0-xr)*yr;
            dT[(idx+1) + (idy+1)*Nddf] += g*xr*yr;
        }
    }
}


// the adjoint of the sweeps and of the seeding: from {lambda}, the gradient with respect to the swept
// arrival times, the gradients with respect to the rupture velocities and the hypocenter into {grad};
// the sweeps backwards, the inputs of each update rebuilt from the snapshots; {lambda} is overwritten
template <typename TYPE>
void
altar::models::seismic::Kinematic<TYPE>::
_sweepAdjoint(const TYPE * M, const TYPE * snapshots, TYPE * lambda, TYPE * grad) const
{
    const int Nasf = _Nasf, Nddf = _Nddf, cells = Nasf*Nddf, Npatch = _Npatch;
    const int Nas = _Nas, Ndd = _Ndd, Nmesh = _Nmesh;
    const TYPE h = _h;
    std::vector<TYPE> gvr(Npatch + 2, TYPE(0));

    for (int sweep = 4*_sweeps - 1; sweep >= 0; --sweep) {
        const int s = sweep % 4;
        const TYPE * before = snapshots + sweep*cells;
        const TYPE * after = before + cells;
        for (int k = cells - 1; k >= 0; --k) {
            int i, j;
            cell(s, k, Nasf, Nddf, i, j);
            const int c = i*Nddf + j;
            const int here = order(s, i, j);
            // the neighbors, and what the update read of each
            int n[4] = { std::max(0,i-1)*Nddf + j, std::min(Nasf-1,i+1)*Nddf + j,
                         i*Nddf + std::max(0,j-1), i*Nddf + std::min(Nddf-1,j+1) };
            int ni[4] = { std::max(0,i-1), std::min(Nasf-1,i+1), i, i };
            int nj[4] = { j, j, std::max(0,j-1), std::min(Nddf-1,j+1) };
            TYPE u[4];
            for (int m = 0; m < 4; ++m) {
                u[m] = n[m] == c ? before[c] : (order(s, ni[m], nj[m]) < here ? after[n[m]] : before[n[m]]);
            }
            const TYPE u_old = before[c];
            const TYPE ux = std::min(u[0], u[1]);
            const TYPE uy = std::min(u[2], u[3]);
            // the rupture velocity, as in the update
            int pi = i/Nmesh - 1; pi = pi < 0 ? 0 : (pi >= Nas ? Nas-1 : pi);
            int pj = j/Nmesh - 1; pj = pj < 0 ? 0 : (pj >= Ndd ? Ndd-1 : pj);
            const int patch = pi*Ndd + pj;
            const TYPE vr = M[_idx[3*Npatch + patch]];
            const TYPE f = 1./vr;
            const TYPE fh = f*h;
            const TYPE d = ux - uy;
            const bool apart = std::fabs(d) >= fh;
            const TYPE root = apart ? TYPE(1) : TYPE(std::sqrt(2.*f*f*h*h - d*d));
            const TYPE u_new = apart ? std::min(ux, uy) + fh : TYPE((ux + uy + root)/2.0);
            // the gradient with respect to the updated value, split between the update and the old value
            const TYPE g = lambda[c];
            const TYPE g_new = dmin(u_new, u_old, g);
            TYPE gux, guy, gfh;
            if (apart) {
                gux = dmin(ux, uy, g_new);
                guy = dmin(uy, ux, g_new);
                gfh = g_new;
            } else {
                gux = g_new*0.5*(1 - d/root);
                guy = g_new*0.5*(1 + d/root);
                gfh = g_new*fh/root;
            }
            TYPE gu[4] = { dmin(u[0], u[1], gux), dmin(u[1], u[0], gux),
                           dmin(u[2], u[3], guy), dmin(u[3], u[2], guy) };
            // the cell now carries the gradient with respect to its value before the update
            TYPE g_old = dmin(u_old, u_new, g);
            for (int m = 0; m < 4; ++m) {
                if (n[m] == c) g_old += gu[m];
                else lambda[n[m]] += gu[m];
            }
            lambda[c] = g_old;
            // fh = h/vr
            gvr[patch] += -gfh*h/(vr*vr);
        }
    }

    // the seeding: T = min((distance + steepness far^2/h)/vr, cap), far = max(0, distance - radius h)
    const TYPE hypo_strike = M[_idx[4*Npatch]] + _dsp*1.5;
    const TYPE hypo_dip = M[_idx[4*Npatch+1]] + _dsp*1.5;
    for (int c = 0; c < cells; ++c) {
        const int i = c/Nddf, j = c%Nddf;
        const TYPE ds = (i+0.5)*h - hypo_strike;
        const TYPE dd = (j+0.5)*h - hypo_dip;
        const TYPE distance = std::sqrt(ds*ds + dd*dd);
        const TYPE excess = distance - _radius*h;
        const TYPE far = std::max(TYPE(0), excess);
        int pi = std::min(std::max(i/Nmesh - 1, 0), Nas - 1);
        int pj = std::min(std::max(j/Nmesh - 1, 0), Ndd - 1);
        const int patch = pi*Ndd + pj;
        const TYPE vr = M[_idx[3*Npatch + patch]];
        const TYPE travel = (distance + _steepness*far*far/h)/vr;
        const TYPE g = dmin(travel, _cap, lambda[c]);
        if (g == 0) continue;
        gvr[patch] += -g*travel/vr;
        // the distance, through the hypocenter; none at the hypocenter itself
        const TYPE dfar = excess > 0 ? TYPE(1) : (excess == 0 ? TYPE(0.5) : TYPE(0));
        const TYPE gdistance = g*(1 + 2*_steepness*far*dfar/h)/vr;
        if (distance > 0) {
            gvr[Npatch] += -gdistance*ds/distance;
            gvr[Npatch+1] += -gdistance*dd/distance;
        }
    }

    for (int p = 0; p < Npatch; ++p) grad[_idx[3*Npatch + p]] = gvr[p];
    grad[_idx[4*Npatch]] = gvr[Npatch];
    grad[_idx[4*Npatch+1]] = gvr[Npatch+1];
}


// explicit instantiation
template class altar::models::seismic::Kinematic<float>;
template class altar::models::seismic::Kinematic<double>;

// end of file
