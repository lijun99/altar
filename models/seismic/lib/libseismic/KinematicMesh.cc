// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//

// the kinematic model on a triangulated fault on the cpu, the arithmetic of
// cudaKinematicMesh_kernels.cu; the fast iterative method updates the active vertices of a pass in
// order, in place, and converges to the arrival times of the cuda one, which updates them in parallel

#include "KinematicMesh.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace {
    // the upwind update of C from the edge AB of a triangle of slowness {s}: the time, and its
    // derivatives with respect to TA, TB and s, by the envelope theorem the interpolation weights
    // of the point the ray crosses AB at, and the length of its path to C
    template <typename TYPE>
    TYPE update(TYPE TA, TYPE TB, const TYPE * A, const TYPE * B, const TYPE * C, TYPE s,
                TYPE & wA, TYPE & wB, TYPE & ws)
    {
        TYPE w[3], e[3], f[3];
        for (int k = 0; k < 3; ++k) { w[k] = C[k] - A[k]; e[k] = B[k] - A[k]; f[k] = C[k] - B[k]; }
        TYPE a = e[0]*e[0] + e[1]*e[1] + e[2]*e[2];
        TYPE b = w[0]*e[0] + w[1]*e[1] + w[2]*e[2];
        TYPE c = w[0]*w[0] + w[1]*w[1] + w[2]*w[2];
        TYPE D = std::max(a*c - b*b, TYPE(0));
        // the one-point updates, from A and from B
        TYPE lA = std::sqrt(c), lB = std::sqrt(f[0]*f[0] + f[1]*f[1] + f[2]*f[2]);
        TYPE tA = TA + s*lA, tB = TB + s*lB;
        TYPE T;
        if (tA <= tB) { T = tA; wA = 1; wB = 0; ws = lA; }
        else { T = tB; wA = 0; wB = 1; ws = lB; }
        // the two-point update, where the ray crosses the inside of AB
        TYPE k = (TB - TA)/s;
        if (k*k < a && std::isfinite(k)) {
            TYPE mu = k*std::sqrt(D/(a - k*k));
            TYPE lambda = (b - mu)/a;
            TYPE length = std::sqrt((mu*mu + D)/a);
            TYPE t2 = TA + lambda*(TB - TA) + s*length;
            if (lambda > 0 && lambda < 1 && t2 < T) { T = t2; wA = 1 - lambda; wB = lambda; ws = length; }
        }
        return T;
    }

    // the seed of the vertex at {x}: the straight ray time from the hypocenter {hypo} (its point,
    // then its 3 x 2 jacobian), within {radius} spacings, rising steeply past it, capped; and its
    // derivatives with respect to the hypocenter and the slowness {s}
    template <typename TYPE>
    TYPE seed(const TYPE * x, const TYPE * hypo, TYPE s, TYPE spacing, TYPE radius,
              TYPE steepness, TYPE cap, TYPE * dh, TYPE & ds)
    {
        TYPE r[3] = {x[0] - hypo[0], x[1] - hypo[1], x[2] - hypo[2]};
        TYPE distance = std::sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
        TYPE far = std::max(TYPE(0), distance - radius*spacing);
        TYPE path = distance + steepness*far*far/spacing;
        TYPE T0 = path*s;
        if (T0 >= cap) { dh[0] = dh[1] = 0; ds = 0; return cap; }
        TYPE dpath = 1 + 2*steepness*far/spacing;
        TYPE inv = distance > 0 ? 1/distance : TYPE(0);
        const TYPE * J = hypo + 3;
        for (int c = 0; c < 2; ++c) {
            TYPE dd = -(r[0]*J[0*2+c] + r[1]*J[1*2+c] + r[2]*J[2*2+c])*inv;
            dh[c] = s*dpath*dd;
        }
        ds = path;
        return T0;
    }
}


// constructor; copies the mesh tables
template <typename TYPE>
altar::models::seismic::KinematicMesh<TYPE>::
KinematicMesh(std::size_t Np, std::size_t Nt, double dt, const double * t0s,
              std::size_t V, const double * xyz, std::size_t E, const int * incPtr,
              const int * incA, const int * incB, const int * incPatch, std::size_t N,
              const int * nbPtr, const int * nb, std::size_t VP, const int * vpPtr,
              const int * vpPatch, const double * patchGeometry, double spacing,
              std::size_t P, std::size_t Nsp, const int * corners, const double * weights,
              const std::size_t * idx_map) :
    _Np(Np), _Nt(Nt), _Nsp(Nsp), _P(P), _V(V), _NGb(2*Np*Nt), _dt(dt), _spacing(spacing),
    // a vertex changes at most once per pass of the method along its longest upwind path
    _maxiter(int(4*V + 16)),
    _t0s(t0s, t0s + Np), _xyz(xyz, xyz + 3*V), _patchGeometry(patchGeometry, patchGeometry + 15*Np),
    _weights(weights, weights + 3*P),
    _incPtr(incPtr, incPtr + V + 1), _incA(incA, incA + E), _incB(incB, incB + E),
    _incPatch(incPatch, incPatch + E), _nbPtr(nbPtr, nbPtr + V + 1), _nb(nb, nb + N),
    _vpPtr(vpPtr, vpPtr + V + 1), _vpPatch(vpPatch, vpPatch + VP), _corners(corners, corners + 3*P),
    _idx(idx_map, idx_map + 4*Np + 2)
{}


// Mb for each sample
template <typename TYPE>
void
altar::models::seismic::KinematicMesh<TYPE>::
calculateBigM(const TYPE * theta, TYPE * Mb, std::size_t parameters, std::size_t batch) const
{
    _iterations.assign(batch, 0);
    #pragma omp parallel
    {
        std::vector<TYPE> T(_V), TI0(_P);
        std::vector<char> cur(_V), nxt(_V);
        TYPE H[9];
        #pragma omp for schedule(dynamic)
        for (long sample = 0; sample < long(batch); ++sample) {
            const TYPE * M = theta + sample*parameters;
            _hypocenter(M, H);
            _iterations[sample] = _eikonal(M, H, T.data(), cur, nxt);
            _interpolate(T.data(), TI0.data());
            _cast(M, TI0.data(), Mb + sample*_NGb);
        }
    }
}


// the arrival times for each sample
template <typename TYPE>
void
altar::models::seismic::KinematicMesh<TYPE>::
arrivals(const TYPE * theta, TYPE * T, std::size_t parameters, std::size_t batch) const
{
    _iterations.assign(batch, 0);
    #pragma omp parallel
    {
        std::vector<char> cur(_V), nxt(_V);
        TYPE H[9];
        #pragma omp for schedule(dynamic)
        for (long sample = 0; sample < long(batch); ++sample) {
            const TYPE * M = theta + sample*parameters;
            _hypocenter(M, H);
            _iterations[sample] = _eikonal(M, H, T + sample*_V, cur, nxt);
        }
    }
}


// the gradient: the arrival times again, then the adjoints of the source time functions, the
// interpolation and the eikonal
template <typename TYPE>
void
altar::models::seismic::KinematicMesh<TYPE>::
gradient(const TYPE * theta, const TYPE * dMb, TYPE * grad, std::size_t parameters,
         std::size_t batch) const
{
    _iterations.assign(batch, 0);
    #pragma omp parallel
    {
        std::vector<TYPE> T(_V), dT(_V), TI0(_P), dTI0(_P), weight(3*_V), gs(_Np);
        std::vector<int> order(_V), parent(3*_V);
        std::vector<char> cur(_V), nxt(_V);
        TYPE H[9];
        #pragma omp for schedule(dynamic)
        for (long sample = 0; sample < long(batch); ++sample) {
            const TYPE * M = theta + sample*parameters;
            TYPE * g = grad + sample*parameters;
            _hypocenter(M, H);
            _iterations[sample] = _eikonal(M, H, T.data(), cur, nxt);
            _interpolate(T.data(), TI0.data());
            _castAdjoint(M, TI0.data(), dMb + sample*_NGb, g, dTI0.data());
            // the adjoint of the interpolation
            std::fill(dT.begin(), dT.end(), TYPE(0));
            for (std::size_t q = 0; q < _P; ++q) {
                for (int k = 0; k < 3; ++k) dT[_corners[3*q+k]] += dTI0[q]*_weights[3*q+k];
            }
            _eikonalAdjoint(M, H, T.data(), dT.data(), g, order, parent, weight, gs);
        }
    }
}


// the most passes of the fast iterative method
template <typename TYPE>
int
altar::models::seismic::KinematicMesh<TYPE>::
iterations(std::size_t batch) const
{
    std::size_t n = std::min(batch, _iterations.size());
    return n ? *std::max_element(_iterations.begin(), _iterations.begin() + n) : 0;
}


// the hypocenter: the patch whose triangle holds it in surface coordinates, or the nearest, and the
// affine map of the patch to the fault; {H} gets the point (3) and the jacobian (6)
template <typename TYPE>
void
altar::models::seismic::KinematicMesh<TYPE>::
_hypocenter(const TYPE * M, TYPE * H) const
{
    const int Np = _Np;
    TYPE hu = M[_idx[4*Np]], hv = M[_idx[4*Np+1]];
    int best = 0;
    TYPE bestWorst = -1e30;
    for (int p = 0; p < Np; ++p) {
        const TYPE * g = _patchGeometry.data() + 15*p;
        TYPE v0u = g[2] - g[0], v0v = g[3] - g[1], v1u = g[4] - g[0], v1v = g[5] - g[1];
        TYPE du = hu - g[0], dv = hv - g[1];
        TYPE den = v0u*v1v - v0v*v1u;
        TYPE l1 = (du*v1v - dv*v1u)/den, l2 = (v0u*dv - v0v*du)/den;
        TYPE worst = std::min(std::min(1 - l1 - l2, l1), l2);
        if (worst > bestWorst) { bestWorst = worst; best = p; }
    }
    const TYPE * g = _patchGeometry.data() + 15*best;
    TYPE du = hu - g[0], dv = hv - g[1];
    for (int r = 0; r < 3; ++r) H[r] = g[6+r] + g[9+2*r]*du + g[9+2*r+1]*dv;
    for (int k = 0; k < 6; ++k) H[3+k] = g[9+k];
}


// the arrival times at the vertices: the seeds near the hypocenter, within two spacings past the
// seeding radius of it or of the closest vertex, the others at the cap, then passes of the fast
// iterative method until one changes nothing; returns the passes
template <typename TYPE>
int
altar::models::seismic::KinematicMesh<TYPE>::
_eikonal(const TYPE * M, const TYPE * H, TYPE * T, std::vector<char> & cur,
         std::vector<char> & nxt) const
{
    const int Np = _Np, V = _V;
    const TYPE * X = _xyz.data();
    std::vector<TYPE> s_patch(Np);
    for (int p = 0; p < Np; ++p) s_patch[p] = 1/M[_idx[3*Np + p]];
    std::fill(cur.begin(), cur.end(), 0);
    std::fill(nxt.begin(), nxt.end(), 0);
    // the distance of the closest vertex to the hypocenter
    TYPE closest = _cap;
    for (int v = 0; v < V; ++v) {
        TYPE r0 = X[3*v] - H[0], r1 = X[3*v+1] - H[1], r2 = X[3*v+2] - H[2];
        closest = std::min(closest, std::sqrt(r0*r0 + r1*r1 + r2*r2));
    }
    const TYPE near = std::max(_radius + 2, closest/_spacing + 2)*_spacing;
    for (int v = 0; v < V; ++v) {
        TYPE r0 = X[3*v] - H[0], r1 = X[3*v+1] - H[1], r2 = X[3*v+2] - H[2];
        if (std::sqrt(r0*r0 + r1*r1 + r2*r2) > near) { T[v] = _cap; continue; }
        // the slowness a vertex seeds with, the least of its patches'
        TYPE s = s_patch[_vpPatch[_vpPtr[v]]];
        for (int e = _vpPtr[v] + 1; e < _vpPtr[v+1]; ++e) s = std::min(s, s_patch[_vpPatch[e]]);
        TYPE dh[2], ds;
        T[v] = seed(X + 3*v, H, s, _spacing, _radius, _steepness, _cap, dh, ds);
        cur[v] = 1;
    }

    int iteration = 0;
    while (iteration < _maxiter) {
        ++iteration;
        bool changed = false;
        for (int v = 0; v < V; ++v) {
            if (!cur[v]) continue;
            TYPE t = T[v];
            TYPE best = t;
            for (int e = _incPtr[v]; e < _incPtr[v+1]; ++e) {
                int a = _incA[e], b = _incB[e];
                TYPE wA, wB, ws;
                best = std::min(best, update<TYPE>(T[a], T[b], X + 3*a, X + 3*b, X + 3*v,
                                                   s_patch[_incPatch[e]], wA, wB, ws));
            }
            if (best < t) {
                T[v] = best;
                changed = true;
                for (int e = _nbPtr[v]; e < _nbPtr[v+1]; ++e) nxt[_nb[e]] = 1;
            }
        }
        if (!changed) break;
        std::swap(cur, nxt);
        std::fill(nxt.begin(), nxt.end(), 0);
    }
    return iteration;
}


// the arrival times at the source points, by barycentric interpolation in their triangles
template <typename TYPE>
void
altar::models::seismic::KinematicMesh<TYPE>::
_interpolate(const TYPE * T, TYPE * TI0) const
{
    for (std::size_t q = 0; q < _P; ++q) {
        TI0[q] = _weights[3*q]*T[_corners[3*q]] + _weights[3*q+1]*T[_corners[3*q+1]]
            + _weights[3*q+2]*T[_corners[3*q+2]];
    }
}


// the slips of each patch over time: the triangles of its {Nsp} source points, contiguous, sampled
// at the start of each time interval
template <typename TYPE>
void
altar::models::seismic::KinematicMesh<TYPE>::
_cast(const TYPE * M, const TYPE * TI0, TYPE * Mb) const
{
    const int Np = _Np, Nsp = _Nsp;
    for (int patch = 0; patch < Np; ++patch) {
        const TYPE * TI = TI0 + patch*Nsp;
        TYPE ss = M[_idx[patch]], sd = M[_idx[Np + patch]];
        TYPE hTr = M[_idx[2*Np + patch]]/2;
        for (int i = 0; i < int(_Nt); ++i) {
            TYPE t0 = _t0s[patch] + _dt*i;
            TYPE c = 0;
            for (int q = 0; q < Nsp; ++q) {
                TYPE t = TI[q] + hTr;
                if (t0 <= t - hTr || t0 >= t + hTr) continue;
                c += (1 - std::abs(t - t0)/hTr)/hTr/Nsp;
            }
            Mb[i*2*Np + patch] = c*ss;
            Mb[i*2*Np + Np + patch] = c*sd;
        }
    }
}


// the adjoint of the source time functions
template <typename TYPE>
void
altar::models::seismic::KinematicMesh<TYPE>::
_castAdjoint(const TYPE * M, const TYPE * TI0, const TYPE * dMb, TYPE * grad, TYPE * dTI0) const
{
    const int Np = _Np, Nsp = _Nsp;
    for (int patch = 0; patch < Np; ++patch) {
        const TYPE * TI = TI0 + patch*Nsp;
        TYPE * dTI = dTI0 + patch*Nsp;
        TYPE ss = M[_idx[patch]], sd = M[_idx[Np + patch]];
        TYPE hTr = M[_idx[2*Np + patch]]/2;
        for (int q = 0; q < Nsp; ++q) dTI[q] = 0;
        TYPE gss = 0, gsd = 0, ghTr = 0;
        for (int i = 0; i < int(_Nt); ++i) {
            TYPE t0 = _t0s[patch] + _dt*i;
            TYPE dss = dMb[i*2*Np + patch], dsd = dMb[i*2*Np + Np + patch];
            TYPE gc = dss*ss + dsd*sd;
            TYPE c = 0;
            for (int q = 0; q < Nsp; ++q) {
                TYPE t = TI[q] + hTr;
                if (t0 <= t - hTr || t0 >= t + hTr) continue;
                TYPE a = t - t0;
                TYPE absa = std::abs(a);
                TYPE sign = TYPE((a > 0) - (a < 0));
                c += (1 - absa/hTr)/hTr/Nsp;
                // v = (1 - |a|/h)/h/Nsp, with a = TI0 + h - t
                TYPE dv_da = -sign/(hTr*hTr*Nsp);
                TYPE dv_dh = (2*absa/hTr - 1)/(hTr*hTr*Nsp);
                dTI[q] += gc*dv_da;
                ghTr += gc*(dv_dh + dv_da);
            }
            gss += dss*c;
            gsd += dsd*c;
        }
        grad[_idx[patch]] = gss;
        grad[_idx[Np + patch]] = gsd;
        grad[_idx[2*Np + patch]] = ghTr/2;
    }
}


// the adjoint of the eikonal: the update each vertex took at the converged arrival times, then the
// vertices latest first, each handing its adjoint to its parents or, if seeded, to the hypocenter
// and its slowness; then the slowness gradients to the rupture velocities, accumulated, since
// patches may share a velocity column; {dT} holds the adjoint of the arrival times on entry
template <typename TYPE>
void
altar::models::seismic::KinematicMesh<TYPE>::
_eikonalAdjoint(const TYPE * M, const TYPE * H, const TYPE * T, TYPE * dT, TYPE * grad,
                std::vector<int> & order, std::vector<int> & parent, std::vector<TYPE> & weight,
                std::vector<TYPE> & gs) const
{
    const int Np = _Np, V = _V;
    const TYPE * X = _xyz.data();
    // the parents: two upwind vertices and the patch, or -1 and the patch of the seed's slowness
    for (int v = 0; v < V; ++v) {
        TYPE best = 1e30;
        int bestA = -1, bestB = -1, bestPatch = -1;
        TYPE bestWA = 0, bestWB = 0, bestWS = 0;
        for (int e = _incPtr[v]; e < _incPtr[v+1]; ++e) {
            int a = _incA[e], b = _incB[e], p = _incPatch[e];
            TYPE wA, wB, ws;
            TYPE u = update<TYPE>(T[a], T[b], X + 3*a, X + 3*b, X + 3*v, 1/M[_idx[3*Np + p]], wA, wB, ws);
            if (u < best) { best = u; bestA = a; bestB = b; bestPatch = p; bestWA = wA; bestWB = wB; bestWS = ws; }
        }
        int owner = _vpPatch[_vpPtr[v]];
        TYPE s = 1/M[_idx[3*Np + owner]];
        for (int e = _vpPtr[v] + 1; e < _vpPtr[v+1]; ++e) {
            TYPE t = 1/M[_idx[3*Np + _vpPatch[e]]];
            if (t < s) { s = t; owner = _vpPatch[e]; }
        }
        TYPE dh[2], ds;
        TYPE T0 = seed(X + 3*v, H, s, _spacing, _radius, _steepness, _cap, dh, ds);
        int * pv = parent.data() + 3*v;
        TYPE * wv = weight.data() + 3*v;
        if (T0 <= best) {
            pv[0] = -1; pv[1] = -1; pv[2] = owner;
            wv[0] = dh[0]; wv[1] = dh[1]; wv[2] = ds;
        } else {
            pv[0] = bestA; pv[1] = bestB; pv[2] = bestPatch;
            wv[0] = bestWA; wv[1] = bestWB; wv[2] = bestWS;
        }
    }
    // the vertices, latest first
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [T](int a, int b) { return T[a] > T[b]; });
    std::fill(gs.begin(), gs.end(), TYPE(0));
    TYPE gh0 = 0, gh1 = 0;
    for (int k = 0; k < V; ++k) {
        int v = order[k];
        TYPE l = dT[v];
        if (l == 0) continue;
        const int * pv = parent.data() + 3*v;
        const TYPE * wv = weight.data() + 3*v;
        if (pv[0] < 0) {
            gh0 += l*wv[0];
            gh1 += l*wv[1];
        } else {
            dT[pv[0]] += wv[0]*l;
            dT[pv[1]] += wv[1]*l;
        }
        gs[pv[2]] += wv[2]*l;
    }
    for (int p = 0; p < Np; ++p) grad[_idx[3*Np + p]] = 0;
    for (int p = 0; p < Np; ++p) {
        TYPE vr = M[_idx[3*Np + p]];
        grad[_idx[3*Np + p]] += -gs[p]/(vr*vr);
    }
    grad[_idx[4*Np]] = gh0;
    grad[_idx[4*Np+1]] = gh1;
}


// explicit instantiation
template class altar::models::seismic::KinematicMesh<float>;
template class altar::models::seismic::KinematicMesh<double>;

// end of file
