// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//

#include "cudaKinematicMesh_kernels.h"

// the parameters of a sample, in {gIdx} order: the strike slips, dip slips, rise times and rupture
// velocities of the {Np} patches, then the hypocenter along strike and down dip
namespace cudaKinematicMesh_kernels {

    // the upwind update of C from the edge AB of a triangle of slowness {s}: the time, and its
    // derivatives with respect to TA, TB and s, by the envelope theorem the interpolation weights
    // of the point the ray crosses AB at, and the length of its path to C
    template <typename TYPE>
    __device__ TYPE update(TYPE TA, TYPE TB, const TYPE * A, const TYPE * B, const TYPE * C, TYPE s,
                           TYPE & wA, TYPE & wB, TYPE & ws)
    {
        TYPE w[3], e[3], f[3];
        for (int k = 0; k < 3; ++k) { w[k] = C[k] - A[k]; e[k] = B[k] - A[k]; f[k] = C[k] - B[k]; }
        TYPE a = e[0]*e[0] + e[1]*e[1] + e[2]*e[2];
        TYPE b = w[0]*e[0] + w[1]*e[1] + w[2]*e[2];
        TYPE c = w[0]*w[0] + w[1]*w[1] + w[2]*w[2];
        TYPE D = max(a*c - b*b, TYPE(0));
        // the one-point updates, from A and from B
        TYPE lA = sqrt(c), lB = sqrt(f[0]*f[0] + f[1]*f[1] + f[2]*f[2]);
        TYPE tA = TA + s*lA, tB = TB + s*lB;
        TYPE T;
        if (tA <= tB) { T = tA; wA = 1; wB = 0; ws = lA; }
        else { T = tB; wA = 0; wB = 1; ws = lB; }
        // the two-point update, where the ray crosses the inside of AB
        TYPE k = (TB - TA)/s;
        if (k*k < a && isfinite(k)) {
            TYPE mu = k*sqrt(D/(a - k*k));
            TYPE lambda = (b - mu)/a;
            TYPE length = sqrt((mu*mu + D)/a);
            TYPE t2 = TA + lambda*(TB - TA) + s*length;
            if (lambda > 0 && lambda < 1 && t2 < T) { T = t2; wA = 1 - lambda; wB = lambda; ws = length; }
        }
        return T;
    }

    // the seed of the vertex at {x}: the straight ray time from the hypocenter {hypo} (its point,
    // then its 3 x 2 jacobian), within {radius} spacings, rising steeply past it, capped; and its
    // derivatives with respect to the hypocenter and the slowness {s}
    template <typename TYPE>
    __device__ TYPE seed(const TYPE * x, const TYPE * hypo, TYPE s, TYPE spacing, TYPE radius,
                         TYPE steepness, TYPE cap, TYPE * dh, TYPE & ds)
    {
        TYPE r[3] = {x[0] - hypo[0], x[1] - hypo[1], x[2] - hypo[2]};
        TYPE distance = sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
        TYPE far = max(TYPE(0), distance - radius*spacing);
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

    // the slowness a vertex seeds with, the least of its patches', and the patch that has it
    template <typename TYPE>
    __device__ TYPE seedSlowness(const TYPE * s_patch, const int * gVPPtr, const int * gVPPatch, int v,
                                 int & owner)
    {
        TYPE s = s_patch[gVPPatch[gVPPtr[v]]];
        owner = gVPPatch[gVPPtr[v]];
        for (int e = gVPPtr[v] + 1; e < gVPPtr[v+1]; ++e) {
            TYPE t = s_patch[gVPPatch[e]];
            if (t < s) { s = t; owner = gVPPatch[e]; }
        }
        return s;
    }
}


// the hypocenter of each sample: the patch whose triangle holds it in surface coordinates, or the
// nearest, and the affine map of the patch to the fault; {gPatchGeometry} holds, for each patch,
// the surface coordinates of its corners (6), the point of its first corner (3) and the 3 x 2
// jacobian of the map (6); {gHypo} gets the point (3) and the jacobian (6) of each sample
template <typename TYPE>
__global__ void
cudaKinematicMesh_kernels::
hypocenter_batched(const size_t * gIdx, const TYPE * gM, size_t Nparam, int Np,
    const TYPE * gPatchGeometry, size_t batch, TYPE * gHypo)
{
    size_t sample = threadIdx.x + size_t(blockIdx.x)*blockDim.x;
    if (sample >= batch) return;
    const TYPE * M = gM + sample*Nparam;
    TYPE hu = M[gIdx[4*Np]], hv = M[gIdx[4*Np+1]];
    int best = 0;
    TYPE bestWorst = -1e30;
    for (int p = 0; p < Np; ++p) {
        const TYPE * g = gPatchGeometry + 15*p;
        TYPE v0u = g[2] - g[0], v0v = g[3] - g[1], v1u = g[4] - g[0], v1v = g[5] - g[1];
        TYPE du = hu - g[0], dv = hv - g[1];
        TYPE den = v0u*v1v - v0v*v1u;
        TYPE l1 = (du*v1v - dv*v1u)/den, l2 = (v0u*dv - v0v*du)/den;
        TYPE worst = min(min(1 - l1 - l2, l1), l2);
        if (worst > bestWorst) { bestWorst = worst; best = p; }
    }
    const TYPE * g = gPatchGeometry + 15*best;
    TYPE du = hu - g[0], dv = hv - g[1];
    TYPE * H = gHypo + 9*sample;
    for (int r = 0; r < 3; ++r) H[r] = g[6+r] + g[9+2*r]*du + g[9+2*r+1]*dv;
    for (int k = 0; k < 6; ++k) H[3+k] = g[9+k];
}


// the arrival times on the mesh, one block per sample: the seeds, then the fast iterative method;
// the threads update the active vertices in place, a vertex that changes activates its
// neighbors, until a pass changes nothing; the shared memory holds the slowness of each patch,
// the active vertices as two bitsets and, if {inShared}, the arrival times themselves, copied to
// {gT} at the end. Only the vertices near the hypocenter, within two spacings past the seeding
// radius of it or of the closest vertex, start seeded and active; the others start at the cap,
// since their seeds, steeply above the rupture front, can't win, and the method then only works
// near the front
template <typename TYPE>
__global__ void
cudaKinematicMesh_kernels::
eikonal_batched(const size_t * gIdx, const TYPE * gM, size_t Nparam, int Np, int V,
    const TYPE * gXYZ, const int * gIncPtr, const int * gIncA, const int * gIncB,
    const int * gIncPatch, const int * gNbPtr, const int * gNb, const int * gVPPtr,
    const int * gVPPatch, const TYPE * gHypo, TYPE spacing, TYPE radius, TYPE steepness,
    TYPE cap, int maxiter, bool inShared, TYPE * gT, int * gIterations)
{
    extern __shared__ __align__(sizeof(double)) unsigned char shared[];
    TYPE * s_patch = reinterpret_cast<TYPE *>(shared);
    TYPE * s_T = s_patch + Np;
    const int W = (V + 31)/32;
    unsigned * cur = reinterpret_cast<unsigned *>(inShared ? s_T + V : s_T);
    unsigned * nxt = cur + W;

    const size_t sample = blockIdx.x;
    const TYPE * M = gM + sample*Nparam;
    const TYPE * hypo = gHypo + 9*sample;
    volatile TYPE * T = inShared ? s_T : gT + sample*V;

    __shared__ TYPE s_closest[32];
    for (int p = threadIdx.x; p < Np; p += blockDim.x) s_patch[p] = 1/M[gIdx[3*Np + p]];
    for (int w = threadIdx.x; w < W; w += blockDim.x) { cur[w] = 0; nxt[w] = 0; }
    // the distance of the closest vertex to the hypocenter
    TYPE closest = cap;
    for (int v = threadIdx.x; v < V; v += blockDim.x) {
        const TYPE * x = gXYZ + 3*v;
        TYPE r0 = x[0] - hypo[0], r1 = x[1] - hypo[1], r2 = x[2] - hypo[2];
        closest = min(closest, sqrt(r0*r0 + r1*r1 + r2*r2));
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        closest = min(closest, __shfl_down_sync(0xffffffffu, closest, offset));
    }
    if (threadIdx.x % 32 == 0) s_closest[threadIdx.x / 32] = closest;
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int k = 1; k < (blockDim.x + 31)/32; ++k) s_closest[0] = min(s_closest[0], s_closest[k]);
    }
    __syncthreads();
    const TYPE near = max(radius + 2, s_closest[0]/spacing + 2)*spacing;
    for (int v = threadIdx.x; v < V; v += blockDim.x) {
        const TYPE * x = gXYZ + 3*v;
        TYPE r0 = x[0] - hypo[0], r1 = x[1] - hypo[1], r2 = x[2] - hypo[2];
        if (sqrt(r0*r0 + r1*r1 + r2*r2) > near) { T[v] = cap; continue; }
        int owner;
        TYPE s = seedSlowness(s_patch, gVPPtr, gVPPatch, v, owner);
        TYPE dh[2], ds;
        T[v] = seed(gXYZ + 3*v, hypo, s, spacing, radius, steepness, cap, dh, ds);
        atomicOr(cur + (v >> 5), 1u << (v & 31));
    }
    __syncthreads();

    int iteration = 0;
    while (iteration < maxiter) {
        ++iteration;
        int changed = 0;
        for (int w = threadIdx.x; w < W; w += blockDim.x) {
            unsigned bits = cur[w];
            while (bits) {
                int v = 32*w + __ffs(bits) - 1;
                bits &= bits - 1;
                TYPE t = T[v];
                TYPE best = t;
                for (int e = gIncPtr[v]; e < gIncPtr[v+1]; ++e) {
                    int a = gIncA[e], b = gIncB[e];
                    TYPE wA, wB, ws;
                    TYPE u = update<TYPE>(T[a], T[b], gXYZ + 3*a, gXYZ + 3*b, gXYZ + 3*v,
                                          s_patch[gIncPatch[e]], wA, wB, ws);
                    best = min(best, u);
                }
                if (best < t) {
                    T[v] = best;
                    changed = 1;
                    for (int e = gNbPtr[v]; e < gNbPtr[v+1]; ++e) {
                        int n = gNb[e];
                        atomicOr(nxt + (n >> 5), 1u << (n & 31));
                    }
                }
            }
        }
        if (!__syncthreads_or(changed)) break;
        for (int w = threadIdx.x; w < W; w += blockDim.x) { cur[w] = nxt[w]; nxt[w] = 0; }
        __syncthreads();
    }
    if (inShared) {
        for (int v = threadIdx.x; v < V; v += blockDim.x) gT[sample*V + v] = T[v];
    }
    if (threadIdx.x == 0 && gIterations != nullptr) gIterations[sample] = iteration;
}


// the arrival times at the source points, by barycentric interpolation in their triangles
template <typename TYPE>
__global__ void
cudaKinematicMesh_kernels::
interpolate_batched(const TYPE * gT, int V, const int * gCorners, const TYPE * gWeights, int P,
    size_t batch, TYPE * gTI0)
{
    size_t id = threadIdx.x + size_t(blockIdx.x)*blockDim.x;
    if (id >= batch*P) return;
    size_t sample = id/P;
    int q = id % P;
    const TYPE * T = gT + sample*V;
    gTI0[id] = gWeights[3*q]*T[gCorners[3*q]] + gWeights[3*q+1]*T[gCorners[3*q+1]]
        + gWeights[3*q+2]*T[gCorners[3*q+2]];
}


// the slips of each patch over time, Mb[samples][Nt][2][Np]: the triangles of its {Nsp} source
// points, contiguous, sampled at the start of each time interval
template <typename TYPE>
__global__ void
cudaKinematicMesh_kernels::
castBigM_batched(const size_t * gIdx, const TYPE * gM, size_t Nparam, const TYPE * gTI0,
    const TYPE * gt0s, TYPE dt, int Nt, int Np, int Nsp, size_t batch, TYPE * gMb)
{
    size_t sample = blockIdx.x;
    int patch = threadIdx.y + blockIdx.y*blockDim.y;
    if (sample >= batch || patch >= Np) return;
    const TYPE * M = gM + sample*Nparam;
    const TYPE * TI0 = gTI0 + (sample*Np + patch)*Nsp;
    TYPE * Mb = gMb + sample*Nt*2*Np;
    TYPE ss = M[gIdx[patch]], sd = M[gIdx[Np + patch]];
    TYPE hTr = M[gIdx[2*Np + patch]]/2;
    for (int i = 0; i < Nt; ++i) {
        TYPE t0 = gt0s[patch] + dt*i;
        TYPE c = 0;
        for (int q = 0; q < Nsp; ++q) {
            TYPE TI = TI0[q] + hTr;
            if (t0 <= TI - hTr || t0 >= TI + hTr) continue;
            c += (1 - abs(TI - t0)/hTr)/hTr/Nsp;
        }
        Mb[i*2*Np + patch] = c*ss;
        Mb[i*2*Np + Np + patch] = c*sd;
    }
}


// the adjoint of castBigM
template <typename TYPE>
__global__ void
cudaKinematicMesh_kernels::
castBigM_adjoint_batched(const size_t * gIdx, const TYPE * gM, size_t Nparam, const TYPE * gTI0,
    const TYPE * gdMb, const TYPE * gt0s, TYPE dt, int Nt, int Np, int Nsp, size_t batch,
    TYPE * gGrad, TYPE * gdTI0)
{
    size_t sample = blockIdx.x;
    int patch = threadIdx.y + blockIdx.y*blockDim.y;
    if (sample >= batch || patch >= Np) return;
    const TYPE * M = gM + sample*Nparam;
    const TYPE * TI0 = gTI0 + (sample*Np + patch)*Nsp;
    TYPE * dTI0 = gdTI0 + (sample*Np + patch)*Nsp;
    const TYPE * dMb = gdMb + sample*Nt*2*Np;
    TYPE ss = M[gIdx[patch]], sd = M[gIdx[Np + patch]];
    TYPE hTr = M[gIdx[2*Np + patch]]/2;
    for (int q = 0; q < Nsp; ++q) dTI0[q] = 0;
    TYPE gss = 0, gsd = 0, ghTr = 0;
    for (int i = 0; i < Nt; ++i) {
        TYPE t0 = gt0s[patch] + dt*i;
        TYPE dss = dMb[i*2*Np + patch], dsd = dMb[i*2*Np + Np + patch];
        TYPE gc = dss*ss + dsd*sd;
        TYPE c = 0;
        for (int q = 0; q < Nsp; ++q) {
            TYPE TI = TI0[q] + hTr;
            if (t0 <= TI - hTr || t0 >= TI + hTr) continue;
            TYPE a = TI - t0;
            TYPE absa = abs(a);
            TYPE sign = TYPE((a > 0) - (a < 0));
            c += (1 - absa/hTr)/hTr/Nsp;
            // v = (1 - |a|/h)/h/Nsp, with a = TI0 + h - t
            TYPE dv_da = -sign/(hTr*hTr*Nsp);
            TYPE dv_dh = (2*absa/hTr - 1)/(hTr*hTr*Nsp);
            dTI0[q] += gc*dv_da;
            ghTr += gc*(dv_dh + dv_da);
        }
        gss += dss*c;
        gsd += dsd*c;
    }
    TYPE * grad = gGrad + sample*Nparam;
    grad[gIdx[patch]] = gss;
    grad[gIdx[Np + patch]] = gsd;
    grad[gIdx[2*Np + patch]] = ghTr/2;
}


// the adjoint of the interpolation, one thread a sample; {gdT} must be zeroed first
template <typename TYPE>
__global__ void
cudaKinematicMesh_kernels::
interpolate_adjoint_batched(const TYPE * gdTI0, int P, const int * gCorners, const TYPE * gWeights,
    int V, size_t batch, TYPE * gdT)
{
    size_t sample = threadIdx.x + size_t(blockIdx.x)*blockDim.x;
    if (sample >= batch) return;
    const TYPE * dTI0 = gdTI0 + sample*P;
    TYPE * dT = gdT + sample*V;
    for (int q = 0; q < P; ++q) {
        TYPE g = dTI0[q];
        for (int k = 0; k < 3; ++k) dT[gCorners[3*q+k]] += g*gWeights[3*q+k];
    }
}


// the update each vertex took at the converged arrival times: {gParent} gets, for each vertex, its
// two upwind vertices and its patch, {gWeight} the derivatives of its time with respect to their
// times and the slowness; a seeded vertex gets -1 and the patch it took its slowness from, and the
// derivatives of its seed with respect to the hypocenter and that slowness
template <typename TYPE>
__global__ void
cudaKinematicMesh_kernels::
parents_batched(const size_t * gIdx, const TYPE * gM, size_t Nparam, int Np, int V,
    const TYPE * gXYZ, const int * gIncPtr, const int * gIncA, const int * gIncB,
    const int * gIncPatch, const int * gVPPtr, const int * gVPPatch, const TYPE * gHypo,
    TYPE spacing, TYPE radius, TYPE steepness, TYPE cap, const TYPE * gT, size_t batch,
    int * gParent, TYPE * gWeight)
{
    size_t sample = blockIdx.x;
    int v = threadIdx.x + blockIdx.y*blockDim.x;
    if (sample >= batch || v >= V) return;
    const TYPE * M = gM + sample*Nparam;
    const TYPE * T = gT + sample*V;
    int * parent = gParent + 3*(sample*V + v);
    TYPE * weight = gWeight + 3*(sample*V + v);
    // the best update
    TYPE best = 1e30;
    int bestA = -1, bestB = -1, bestPatch = -1;
    TYPE bestWA = 0, bestWB = 0, bestWS = 0;
    for (int e = gIncPtr[v]; e < gIncPtr[v+1]; ++e) {
        int a = gIncA[e], b = gIncB[e], p = gIncPatch[e];
        TYPE wA, wB, ws;
        TYPE u = update<TYPE>(T[a], T[b], gXYZ + 3*a, gXYZ + 3*b, gXYZ + 3*v,
                              1/M[gIdx[3*Np + p]], wA, wB, ws);
        if (u < best) { best = u; bestA = a; bestB = b; bestPatch = p; bestWA = wA; bestWB = wB; bestWS = ws; }
    }
    // and the seed
    int owner = gVPPatch[gVPPtr[v]];
    TYPE s = 1/M[gIdx[3*Np + owner]];
    for (int e = gVPPtr[v] + 1; e < gVPPtr[v+1]; ++e) {
        TYPE t = 1/M[gIdx[3*Np + gVPPatch[e]]];
        if (t < s) { s = t; owner = gVPPatch[e]; }
    }
    TYPE dh[2], ds;
    TYPE T0 = cudaKinematicMesh_kernels::seed(gXYZ + 3*v, gHypo + 9*sample, s, spacing, radius,
                                              steepness, cap, dh, ds);
    if (T0 <= best) {
        parent[0] = -1; parent[1] = -1; parent[2] = owner;
        weight[0] = dh[0]; weight[1] = dh[1]; weight[2] = ds;
    } else {
        parent[0] = bestA; parent[1] = bestB; parent[2] = bestPatch;
        weight[0] = bestWA; weight[1] = bestWB; weight[2] = bestWS;
    }
}


// the adjoint of the eikonal, one thread a sample: the vertices latest first, {gOrder}, each hands
// its adjoint to its parents, or, if seeded, to the hypocenter and its slowness; then the slowness
// gradients to the rupture velocities; {gdT} holds the adjoint of the arrival times on entry
template <typename TYPE>
__global__ void
cudaKinematicMesh_kernels::
eikonal_adjoint_batched(const size_t * gIdx, const TYPE * gM, size_t Nparam, int Np, int V,
    const int * gOrder, const int * gParent, const TYPE * gWeight, size_t batch, TYPE * gdT,
    TYPE * gGs, TYPE * gGrad)
{
    size_t sample = threadIdx.x + size_t(blockIdx.x)*blockDim.x;
    if (sample >= batch) return;
    const TYPE * M = gM + sample*Nparam;
    const int * order = gOrder + sample*V;
    TYPE * dT = gdT + sample*V;
    TYPE * gs = gGs + sample*Np;
    for (int p = 0; p < Np; ++p) gs[p] = 0;
    TYPE gh0 = 0, gh1 = 0;
    for (int k = 0; k < V; ++k) {
        int v = order[k];
        TYPE l = dT[v];
        if (l == 0) continue;
        const int * parent = gParent + 3*(sample*V + v);
        const TYPE * weight = gWeight + 3*(sample*V + v);
        if (parent[0] < 0) {
            gh0 += l*weight[0];
            gh1 += l*weight[1];
        } else {
            dT[parent[0]] += weight[0]*l;
            dT[parent[1]] += weight[1]*l;
        }
        gs[parent[2]] += weight[2]*l;
    }
    TYPE * grad = gGrad + sample*Nparam;
    for (int p = 0; p < Np; ++p) {
        TYPE vr = M[gIdx[3*Np + p]];
        grad[gIdx[3*Np + p]] = -gs[p]/(vr*vr);
    }
    grad[gIdx[4*Np]] = gh0;
    grad[gIdx[4*Np+1]] = gh1;
}


// end of file
