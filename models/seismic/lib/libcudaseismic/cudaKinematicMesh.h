// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//

// code guard
#if !defined(altar_models_seismic_cudaKinematicMesh_h)
#define altar_models_seismic_cudaKinematicMesh_h

#include <cstddef>
#include <cublas_v2.h>

namespace altar {
    namespace models {
        namespace seismic {
            template<typename TYPE> class cudaKinematicMesh;
        }
    }
}


/// @brief The kinematic model on a triangulated fault: the arrival times of the rupture on the
/// vertices of a mesh refined from the patches, by the fast iterative method, their interpolation
/// to the source points of each patch, the slips of each patch over time, Mb, and the data Gb Mb;
/// with the gradient of the log likelihood by the adjoint of each step
///
/// The mesh tables, copied to the device:
/// - xyz: the vertices, (V x 3), km
/// - incident: for each vertex, the triangles it is a corner of, as CSR (incPtr, V+1), the other
///   two corners (incA, incB) and the patch of each
/// - neighbors: for each vertex, the vertices sharing a triangle with it, as CSR (nbPtr, nb)
/// - vertex patches: for each vertex, the patches of its triangles, as CSR (vpPtr, vpPatch)
/// - patch geometry: for each patch, the surface coordinates of its corners (6), the point of its
///   first corner (3) and the 3 x 2 jacobian of its map from surface coordinates (6)
/// - source points: the corners (P x 3) and the barycentric weights (P x 3) of the triangle of
///   each, {Nsp} consecutive points per patch
template <typename TYPE>
class altar::models::seismic::cudaKinematicMesh
{
public:
    /// the forward model: prediction <- Gb Mb(theta), or Gb Mb(theta) - prediction
    void forwardModel(cublasHandle_t handle, const TYPE * theta, const TYPE * Gb, TYPE * prediction,
        size_t parameters, size_t batch, bool return_residual=true, cudaStream_t stream=0) const;
    /// the slips of each patch over time, Mb[samples][Nt][2][Np]
    void calculateBigM(const TYPE * theta, TYPE * gMb, size_t parameters, size_t batch,
        cudaStream_t stream=0) const;
    /// prediction <- Gb Mb, or Gb Mb - prediction
    void linearBigGM(cublasHandle_t handle, const TYPE * gGb, const TYPE * gMb, TYPE * prediction,
        size_t batch, bool return_residual, cudaStream_t stream=0) const;
    /// the gradient of the log likelihood with respect to my columns of theta, from that with
    /// respect to Mb
    void gradient(const TYPE * theta, const TYPE * gdMb, TYPE * gGrad, size_t parameters,
        size_t batch, cudaStream_t stream=0) const;
    /// the most passes the fast iterative method took over the last {batch} samples
    int iterations(size_t batch) const;

    /// the arrival times on the mesh, of the last forward model
    const TYPE * arrivals() const { return _gpu_T; }

private:
    void _arrivalTimes(const TYPE * theta, size_t parameters, size_t batch, cudaStream_t stream) const;

    // sizes
    size_t _Np, _Nt, _Nsp, _P, _V, _E, _N, _VP;
    size_t _samples, _parameters, _observations, _NGbparameters;
    TYPE _dt, _spacing;
    const TYPE _radius = 1.0; ///< the straight ray seeding radius around the hypocenter, in spacings
    const TYPE _steepness = 10.0; ///< how fast the seeds rise past the radius
    const TYPE _cap = 1.e6; ///< the largest arrival time
    int _maxiter;

    // inputs held by the caller
    const size_t * _gidx_map;
    const TYPE * _gt0s;

    // the mesh tables
    TYPE * _xyz = nullptr;
    int * _incPtr = nullptr, * _incA = nullptr, * _incB = nullptr, * _incPatch = nullptr;
    int * _nbPtr = nullptr, * _nb = nullptr, * _vpPtr = nullptr, * _vpPatch = nullptr;
    TYPE * _patchGeometry = nullptr;
    int * _corners = nullptr;
    TYPE * _weights = nullptr;

    // work
    TYPE * _gpu_Mb = nullptr, * _gpu_T = nullptr, * _gpu_TI0 = nullptr, * _gpu_hypo = nullptr;
    int * _gpu_iterations = nullptr;
    size_t _shared;
    bool _inShared; ///< whether the eikonal keeps the arrival times in shared memory
    // the gradient's, made on its first use
    mutable TYPE * _gpu_dTI0 = nullptr, * _gpu_dT = nullptr, * _gpu_weight = nullptr;
    mutable TYPE * _gpu_keys = nullptr, * _gpu_gs = nullptr;
    mutable int * _gpu_parent = nullptr, * _gpu_order = nullptr, * _gpu_iota = nullptr;
    mutable int * _gpu_offsets = nullptr;
    mutable void * _gpu_sort = nullptr;
    mutable size_t _sortBytes = 0;

public:
    cudaKinematicMesh(size_t Np, size_t Nt, double dt, const TYPE * gt0s,
        size_t V, const double * xyz, size_t E, const int * incPtr, const int * incA,
        const int * incB, const int * incPatch, size_t N, const int * nbPtr, const int * nb,
        size_t VP, const int * vpPtr, const int * vpPatch, const double * patchGeometry,
        double spacing, size_t P, size_t Nsp, const int * corners, const double * weights,
        size_t samples, size_t parameters, size_t observations, const size_t * gidxMap);
    virtual ~cudaKinematicMesh();

private:
    cudaKinematicMesh(const cudaKinematicMesh &) = delete;
    const cudaKinematicMesh & operator=(const cudaKinematicMesh &) = delete;
};

#endif

// end of file
