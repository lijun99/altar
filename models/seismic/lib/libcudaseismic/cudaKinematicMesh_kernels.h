// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//

// code guard
#if !defined(altar_models_seismic_cudaKinematicMesh_kernels_h)
#define altar_models_seismic_cudaKinematicMesh_kernels_h

// the kernels of the kinematic model on a triangulated fault; see cudaKinematicMesh.h for the
// layouts of the mesh tables
namespace cudaKinematicMesh_kernels {

    // the point and the (3 x 2) jacobian of the hypocenter of each sample, from its patch
    template <typename TYPE>
    __global__ void hypocenter_batched(const size_t * gIdx, const TYPE * gM, size_t Nparam,
        int Np, const TYPE * gPatchGeometry, size_t batch, TYPE * gHypo);

    // the arrival times on the mesh: seeded around the hypocenter, then the fast iterative
    // method, one block per sample
    template <typename TYPE>
    __global__ void eikonal_batched(const size_t * gIdx, const TYPE * gM, size_t Nparam, int Np,
        int V, const TYPE * gXYZ, const int * gIncPtr, const int * gIncA, const int * gIncB,
        const int * gIncPatch, const int * gNbPtr, const int * gNb, const int * gVPPtr,
        const int * gVPPatch, const TYPE * gHypo, TYPE spacing, TYPE radius, TYPE steepness,
        TYPE cap, int maxiter, bool inShared, TYPE * gT, int * gIterations);

    // the arrival times at the source points
    template <typename TYPE>
    __global__ void interpolate_batched(const TYPE * gT, int V, const int * gCorners,
        const TYPE * gWeights, int P, size_t batch, TYPE * gTI0);

    // the slips of each patch over time, from the triangles of its source points
    template <typename TYPE>
    __global__ void castBigM_batched(const size_t * gIdx, const TYPE * gM, size_t Nparam,
        const TYPE * gTI0, const TYPE * gt0s, TYPE dt, int Nt, int Np, int Nsp, size_t batch,
        TYPE * gMb);

    // the adjoint of castBigM: the slips and rise times into gGrad, the arrival times at the
    // source points into gdTI0
    template <typename TYPE>
    __global__ void castBigM_adjoint_batched(const size_t * gIdx, const TYPE * gM, size_t Nparam,
        const TYPE * gTI0, const TYPE * gdMb, const TYPE * gt0s, TYPE dt, int Nt, int Np, int Nsp,
        size_t batch, TYPE * gGrad, TYPE * gdTI0);

    // the adjoint of the interpolation: the source points onto the vertices, one thread a sample
    template <typename TYPE>
    __global__ void interpolate_adjoint_batched(const TYPE * gdTI0, int P, const int * gCorners,
        const TYPE * gWeights, int V, size_t batch, TYPE * gdT);

    // the update each vertex took at the converged arrival times, with its derivatives
    template <typename TYPE>
    __global__ void parents_batched(const size_t * gIdx, const TYPE * gM, size_t Nparam, int Np,
        int V, const TYPE * gXYZ, const int * gIncPtr, const int * gIncA, const int * gIncB,
        const int * gIncPatch, const int * gVPPtr, const int * gVPPatch, const TYPE * gHypo,
        TYPE spacing, TYPE radius, TYPE steepness, TYPE cap, const TYPE * gT, size_t batch,
        int * gParent, TYPE * gWeight);

    // the adjoint of the eikonal: the vertices latest first, one thread a sample
    template <typename TYPE>
    __global__ void eikonal_adjoint_batched(const size_t * gIdx, const TYPE * gM, size_t Nparam,
        int Np, int V, const int * gOrder, const int * gParent, const TYPE * gWeight,
        size_t batch, TYPE * gdT, TYPE * gGs, TYPE * gGrad);
}

#endif

// end of file
