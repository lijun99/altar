// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//

// for the build system
#include <portinfo>

// my declarations
#include "cudaKinematicMesh.h"
#include "cudaKinematicMesh_kernels.h"

// my dependencies
#include <algorithm>
#include <type_traits>
#include <vector>
#include <cub/device/device_segmented_sort.cuh>

// shared NTHREADS/IDIVUP/BLOCKDIM/cudaSafeCall/cublasSafeCall
#include <altar/cuda/support.h>


namespace {
    // a device copy of {count} host values, converted to {T}
    template <typename T, typename S>
    T * upload(const S * host, size_t count)
    {
        std::vector<T> values(host, host + count);
        T * device = nullptr;
        cudaSafeCall(cudaMalloc((void **)&device, std::max<size_t>(count, 1)*sizeof(T)));
        if (count) {
            cudaSafeCall(cudaMemcpy(device, values.data(), count*sizeof(T), cudaMemcpyHostToDevice));
        }
        return device;
    }

    // the offsets of the vertices of each sample, and their indices, for the segmented sort
    __global__ void segments(int V, size_t count, int * gOffsets, int * gIota)
    {
        size_t id = threadIdx.x + size_t(blockIdx.x)*blockDim.x;
        if (id <= count) gOffsets[id] = int(id*V);
        for (size_t k = id; k < count*V; k += size_t(gridDim.x)*blockDim.x) gIota[k] = int(k % V);
    }

    // the threads of a block of the eikonal
    const int EIKONAL_THREADS = 256;
}


// constructor
template <typename TYPE>
altar::models::seismic::cudaKinematicMesh<TYPE>::
cudaKinematicMesh(size_t Np, size_t Nt, double dt, const TYPE * gt0s,
    size_t V, const double * xyz, size_t E, const int * incPtr, const int * incA,
    const int * incB, const int * incPatch, size_t N, const int * nbPtr, const int * nb,
    size_t VP, const int * vpPtr, const int * vpPatch, const double * patchGeometry,
    double spacing, size_t P, size_t Nsp, const int * corners, const double * weights,
    size_t samples, size_t parameters, size_t observations, const size_t * gidxMap) :
    _Np(Np), _Nt(Nt), _Nsp(Nsp), _P(P), _V(V), _E(E), _N(N), _VP(VP),
    _samples(samples), _parameters(parameters), _observations(observations),
    _NGbparameters(2*Np*Nt), _dt(dt), _spacing(spacing),
    _gidx_map(gidxMap), _gt0s(gt0s)
{
    // a vertex changes at most once per pass of the method along its longest upwind path
    _maxiter = int(4*V + 16);
    // the mesh tables
    _xyz = upload<TYPE>(xyz, 3*V);
    _incPtr = upload<int>(incPtr, V + 1);
    _incA = upload<int>(incA, E);
    _incB = upload<int>(incB, E);
    _incPatch = upload<int>(incPatch, E);
    _nbPtr = upload<int>(nbPtr, V + 1);
    _nb = upload<int>(nb, N);
    _vpPtr = upload<int>(vpPtr, V + 1);
    _vpPatch = upload<int>(vpPatch, VP);
    _patchGeometry = upload<TYPE>(patchGeometry, 15*Np);
    _corners = upload<int>(corners, 3*P);
    _weights = upload<TYPE>(weights, 3*P);
    // the work arrays
    cudaSafeCall(cudaMalloc((void **)&_gpu_Mb, _NGbparameters*samples*sizeof(TYPE)));
    cudaSafeCall(cudaMalloc((void **)&_gpu_T, V*samples*sizeof(TYPE)));
    cudaSafeCall(cudaMalloc((void **)&_gpu_TI0, P*samples*sizeof(TYPE)));
    cudaSafeCall(cudaMalloc((void **)&_gpu_hypo, 9*samples*sizeof(TYPE)));
    cudaSafeCall(cudaMalloc((void **)&_gpu_iterations, samples*sizeof(int)));
    // the shared memory of the eikonal: the slowness of each patch, the arrival times if they fit,
    // and two bitsets
    int device, most;
    cudaSafeCall(cudaGetDevice(&device));
    cudaSafeCall(cudaDeviceGetAttribute(&most, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    size_t bitsets = 2*((V + 31)/32)*sizeof(unsigned);
    _inShared = (Np + V)*sizeof(TYPE) + bitsets <= size_t(most);
    _shared = (Np + (_inShared ? V : 0))*sizeof(TYPE) + bitsets;
    if (_shared > 48*1024) {
        cudaSafeCall(cudaFuncSetAttribute(cudaKinematicMesh_kernels::eikonal_batched<TYPE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, int(_shared)));
    }
}


// destructor
template <typename TYPE>
altar::models::seismic::cudaKinematicMesh<TYPE>::
~cudaKinematicMesh()
{
    for (void * p : {(void *)_xyz, (void *)_incPtr, (void *)_incA, (void *)_incB, (void *)_incPatch,
                     (void *)_nbPtr, (void *)_nb, (void *)_vpPtr, (void *)_vpPatch,
                     (void *)_patchGeometry, (void *)_corners, (void *)_weights,
                     (void *)_gpu_Mb, (void *)_gpu_T, (void *)_gpu_TI0, (void *)_gpu_hypo,
                     (void *)_gpu_iterations, (void *)_gpu_dTI0, (void *)_gpu_dT,
                     (void *)_gpu_weight, (void *)_gpu_keys, (void *)_gpu_gs, (void *)_gpu_parent,
                     (void *)_gpu_order, (void *)_gpu_iota, (void *)_gpu_offsets, _gpu_sort}) {
        if (p) cudaFree(p);
    }
}


// the arrival times on the mesh
template <typename TYPE>
void
altar::models::seismic::cudaKinematicMesh<TYPE>::
_arrivalTimes(const TYPE * theta, size_t parameters, size_t batch, cudaStream_t stream) const
{
    cudaKinematicMesh_kernels::hypocenter_batched<TYPE>
        <<<IDIVUP(batch, BLOCKDIM), BLOCKDIM, 0, stream>>>
        (_gidx_map, theta, parameters, int(_Np), _patchGeometry, batch, _gpu_hypo);
    cudaSafeCall(cudaGetLastError());
    cudaKinematicMesh_kernels::eikonal_batched<TYPE>
        <<<batch, EIKONAL_THREADS, _shared, stream>>>
        (_gidx_map, theta, parameters, int(_Np), int(_V), _xyz, _incPtr, _incA, _incB, _incPatch,
         _nbPtr, _nb, _vpPtr, _vpPatch, _gpu_hypo, _spacing, _radius, _steepness, _cap, _maxiter,
         _inShared, _gpu_T, _gpu_iterations);
    cudaSafeCall(cudaGetLastError());
    size_t points = batch*_P;
    cudaKinematicMesh_kernels::interpolate_batched<TYPE>
        <<<IDIVUP(points, BLOCKDIM), BLOCKDIM, 0, stream>>>
        (_gpu_T, int(_V), _corners, _weights, int(_P), batch, _gpu_TI0);
    cudaSafeCall(cudaGetLastError());
}


// the slips of each patch over time
template <typename TYPE>
void
altar::models::seismic::cudaKinematicMesh<TYPE>::
calculateBigM(const TYPE * theta, TYPE * gMb, size_t parameters, size_t batch,
    cudaStream_t stream) const
{
    _arrivalTimes(theta, parameters, batch, stream);
    dim3 block(1, BLOCKDIM, 1);
    dim3 grid(batch, IDIVUP(_Np, block.y), 1);
    cudaKinematicMesh_kernels::castBigM_batched<TYPE><<<grid, block, 0, stream>>>
        (_gidx_map, theta, parameters, _gpu_TI0, _gt0s, _dt, int(_Nt), int(_Np), int(_Nsp), batch, gMb);
    cudaSafeCall(cudaGetLastError());
}


// the forward model
template <typename TYPE>
void
altar::models::seismic::cudaKinematicMesh<TYPE>::
forwardModel(cublasHandle_t handle, const TYPE * theta, const TYPE * Gb, TYPE * prediction,
    size_t parameters, size_t batch, bool return_residual, cudaStream_t stream) const
{
    calculateBigM(theta, _gpu_Mb, parameters, batch, stream);
    linearBigGM(handle, Gb, _gpu_Mb, prediction, batch, return_residual, stream);
}


// prediction <- Gb Mb, or Gb Mb - prediction; Gb is (NGbparameters x observations) row-major,
// i.e. (observations x NGbparameters) column-major, as cublas sees it
template <typename TYPE>
void
altar::models::seismic::cudaKinematicMesh<TYPE>::
linearBigGM(cublasHandle_t handle, const TYPE * gGb, const TYPE * gMb, TYPE * prediction,
    size_t batch, bool return_residual, cudaStream_t stream) const
{
    TYPE alpha = 1, beta = return_residual ? -1 : 0;
    if constexpr (std::is_same_v<TYPE, double>) {
        cublasSafeCall(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            _observations, batch, _NGbparameters, &alpha, gGb, _observations,
            gMb, _NGbparameters, &beta, prediction, _observations));
    } else {
        cublasSafeCall(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            _observations, batch, _NGbparameters, &alpha, gGb, _observations,
            gMb, _NGbparameters, &beta, prediction, _observations));
    }
}


// the gradient: the arrival times again, then the adjoints of the source time functions, the
// interpolation and the eikonal
template <typename TYPE>
void
altar::models::seismic::cudaKinematicMesh<TYPE>::
gradient(const TYPE * theta, const TYPE * gdMb, TYPE * gGrad, size_t parameters, size_t batch,
    cudaStream_t stream) const
{
    size_t cells = _V*_samples;
    // the buffers of the gradient, on its first use
    if (_gpu_dT == nullptr) {
        cudaSafeCall(cudaMalloc((void **)&_gpu_dTI0, _P*_samples*sizeof(TYPE)));
        cudaSafeCall(cudaMalloc((void **)&_gpu_dT, cells*sizeof(TYPE)));
        cudaSafeCall(cudaMalloc((void **)&_gpu_weight, 3*cells*sizeof(TYPE)));
        cudaSafeCall(cudaMalloc((void **)&_gpu_parent, 3*cells*sizeof(int)));
        cudaSafeCall(cudaMalloc((void **)&_gpu_keys, cells*sizeof(TYPE)));
        cudaSafeCall(cudaMalloc((void **)&_gpu_order, cells*sizeof(int)));
        cudaSafeCall(cudaMalloc((void **)&_gpu_iota, cells*sizeof(int)));
        cudaSafeCall(cudaMalloc((void **)&_gpu_offsets, (_samples + 1)*sizeof(int)));
        cudaSafeCall(cudaMalloc((void **)&_gpu_gs, _Np*_samples*sizeof(TYPE)));
        segments<<<IDIVUP(std::min<size_t>(cells, 1 << 20), BLOCKDIM), BLOCKDIM>>>
            (int(_V), _samples, _gpu_offsets, _gpu_iota);
        cudaSafeCall(cudaGetLastError());
        cub::DeviceSegmentedSort::SortPairsDescending(nullptr, _sortBytes, _gpu_T, _gpu_keys,
            _gpu_iota, _gpu_order, int(cells), int(_samples), _gpu_offsets, _gpu_offsets + 1);
        cudaSafeCall(cudaMalloc(&_gpu_sort, std::max<size_t>(_sortBytes, 1)));
    }
    // the forward model up to the arrival times at the source points
    _arrivalTimes(theta, parameters, batch, stream);
    // the source time functions: the slips and rise times, and the arrival times at the points
    dim3 block(1, BLOCKDIM, 1);
    dim3 grid(batch, IDIVUP(_Np, block.y), 1);
    cudaKinematicMesh_kernels::castBigM_adjoint_batched<TYPE><<<grid, block, 0, stream>>>
        (_gidx_map, theta, parameters, _gpu_TI0, gdMb, _gt0s, _dt, int(_Nt), int(_Np), int(_Nsp),
         batch, gGrad, _gpu_dTI0);
    cudaSafeCall(cudaGetLastError());
    // the interpolation: the arrival times on the mesh
    cudaSafeCall(cudaMemsetAsync(_gpu_dT, 0, _V*batch*sizeof(TYPE), stream));
    cudaKinematicMesh_kernels::interpolate_adjoint_batched<TYPE>
        <<<IDIVUP(batch, 64), 64, 0, stream>>>
        (_gpu_dTI0, int(_P), _corners, _weights, int(_V), batch, _gpu_dT);
    cudaSafeCall(cudaGetLastError());
    // the update each vertex took
    dim3 pblock(256);
    dim3 pgrid(batch, IDIVUP(_V, pblock.x));
    cudaKinematicMesh_kernels::parents_batched<TYPE><<<pgrid, pblock, 0, stream>>>
        (_gidx_map, theta, parameters, int(_Np), int(_V), _xyz, _incPtr, _incA, _incB, _incPatch,
         _vpPtr, _vpPatch, _gpu_hypo, _spacing, _radius, _steepness, _cap, _gpu_T, batch,
         _gpu_parent, _gpu_weight);
    cudaSafeCall(cudaGetLastError());
    // the vertices of each sample, latest first
    size_t sortBytes = _sortBytes;
    cudaSafeCall(cub::DeviceSegmentedSort::SortPairsDescending(_gpu_sort, sortBytes, _gpu_T,
        _gpu_keys, _gpu_iota, _gpu_order, int(_V*batch), int(batch), _gpu_offsets,
        _gpu_offsets + 1, stream));
    // and back through the eikonal
    cudaKinematicMesh_kernels::eikonal_adjoint_batched<TYPE>
        <<<IDIVUP(batch, 64), 64, 0, stream>>>
        (_gidx_map, theta, parameters, int(_Np), int(_V), _gpu_order, _gpu_parent, _gpu_weight,
         batch, _gpu_dT, _gpu_gs, gGrad);
    cudaSafeCall(cudaGetLastError());
}


// the most passes the eikonal took
template <typename TYPE>
int
altar::models::seismic::cudaKinematicMesh<TYPE>::
iterations(size_t batch) const
{
    std::vector<int> passes(batch);
    cudaSafeCall(cudaMemcpy(passes.data(), _gpu_iterations, batch*sizeof(int), cudaMemcpyDeviceToHost));
    return batch ? *std::max_element(passes.begin(), passes.end()) : 0;
}


// explicit instantiation
template class altar::models::seismic::cudaKinematicMesh<float>;
template class altar::models::seismic::cudaKinematicMesh<double>;

// the kernels, instantiated with the class
#include "cudaKinematicMesh_kernels.cu"

// end of file
