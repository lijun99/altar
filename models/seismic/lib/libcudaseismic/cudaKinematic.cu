// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//
// hailiang zhang
// for the build system
#include <portinfo>

// get my class declaration
#include "cudaKinematic.h"
#include "cudaKinematic_kernels.h"

// my dependencies
#include <pyre/cuda.h>
#include <algorithm>
#include <iostream>

// shared NTHREADS/IDIVUP/BLOCKDIM/cudaSafeCall/cublasSafeCall
#include <altar/cuda/support.h>

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// calculate the forward model
// input theta/M (samples x parameters) ld parameters
// construct a Mb (samples x )
/////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename TYPE>
void
altar::models::seismic::cudaKinematic<TYPE>::
forwardModel(cublasHandle_t handle, const TYPE * const theta, const TYPE * const Gb, TYPE * const prediction,
    const size_t parameters, const size_t batch, bool return_residual, cudaStream_t stream) const
{
    // compute the bigM (gMb)
    calculateBigM(theta, _gpu_Mb, parameters, batch, stream);
    // compute data prediction or residual gRes=gMb*gGb-gObs
    linearBigGM(handle, Gb, _gpu_Mb, prediction, batch, return_residual, stream);
    // all done
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// calculate and return the bigM only
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename TYPE>
void
altar::models::seismic::cudaKinematic<TYPE>::
calculateBigM(const TYPE * const theta, TYPE *const gMb, const size_t parameters,
    const size_t batch, cudaStream_t stream) const
{
    // ... and number of good samples
    int good_samples = batch;
    const TYPE * const gM_candidate_queued = theta;

    // set distance/time for 2x2 mesh grids around hypocenter
    _initT0(gM_candidate_queued, parameters, good_samples, stream);
    // find 4 nearest mesh grids close to hypocenter, set their arrival time
    // set arrival times for all mesh grids
    _fastSweeping(gM_candidate_queued, parameters, good_samples, stream);
    // set arrival time for patches (average over mesh grids, but fine-tuned on time intervals Npt)
    _interpolateT0(good_samples, stream);
    // cast to time dependent slips for patches; gMb[samples][Nt][2(strike,dip slips)][Nas][Ndd]
    _castBigM(gM_candidate_queued, gMb, parameters, good_samples, stream); // where idx_map comes to play
    // all done
}

// the gradient of the log likelihood with respect to my columns of theta, from {gdMb}, that with
// respect to Mb: the forward model again, keeping the arrival times between sweeps, then the
// adjoints of the source time functions, the interpolation, and the sweeps and the seeding
template <typename TYPE>
void
altar::models::seismic::cudaKinematic<TYPE>::
gradient(const TYPE * const theta, const TYPE * const gdMb, TYPE * const gGrad, const size_t parameters,
    const size_t batch, cudaStream_t stream) const
{
    const size_t cells = _Nddf*_Nasf;
    const size_t points = _Npatch*_Npt*_Npt;
    // the buffers of the gradient, on its first use
    if (_gpu_T0_snapshots == nullptr) {
        cudaSafeCall(cudaMalloc((void**)&_gpu_T0_snapshots, (4*_sweep_iter+1)*cells*_samples*sizeof(TYPE)));
        cudaSafeCall(cudaMalloc((void**)&_gpu_dT0, cells*_samples*sizeof(TYPE)));
        cudaSafeCall(cudaMalloc((void**)&_gpu_dTI0, points*_samples*sizeof(TYPE)));
    }
    // the forward model up to the arrival times at the source points
    _initT0(theta, parameters, batch, stream);
    _fastSweeping(theta, parameters, batch, stream, _gpu_T0_snapshots);
    _interpolateT0(batch, stream);
    // the source time functions: the slips and rise times, and the arrival times at the source points
    dim3 cast_block(1, BLOCKDIM, 1);
    dim3 cast_grid(batch, IDIVUP(_Npatch, cast_block.y), 1);
    cudaKinematic_kernels::castBigM_adjoint_batched<TYPE><<<cast_grid, cast_block, 0, stream>>>
        (_gidx_map, theta, _gpu_TI0, gdMb, _gt0s, _dt, parameters, _Nt, _Nas, _Ndd, _Npt, gGrad, _gpu_dTI0);
    cudaSafeCall(cudaGetLastError());
    // the interpolation: the arrival times on the mesh
    cudaSafeCall(cudaMemsetAsync(_gpu_dT0, 0, cells*batch*sizeof(TYPE), stream));
    dim3 interp_grid(IDIVUP(batch, BLOCKDIM)), interp_block(BLOCKDIM);
    cudaKinematic_kernels::interpolateT0_adjoint_batched<TYPE><<<interp_grid, interp_block, 0, stream>>>
        (_gpu_dTI0, _gpu_dT0, batch, _Nas, _Ndd, _Nmesh, _Npt);
    cudaSafeCall(cudaGetLastError());
    // the sweeps and the seeding: the rupture velocities and the hypocenter
    dim3 sweep_grid(batch), sweep_block(_sweepBlock());
    cudaKinematic_kernels::fastSweeping_adjoint_batched<TYPE>
        <<<sweep_grid, sweep_block, (_Npatch+2)*sizeof(TYPE), stream>>>
        (_gidx_map, theta, _gpu_T0_snapshots, _gpu_dT0, parameters, _Nas, _Ndd, _Nmesh, _dsp,
         _seed_radius, _seed_steepness, _it0, _sweep_iter, gGrad);
    cudaSafeCall(cudaGetLastError());
}

// constructor
template <typename TYPE>
altar::models::seismic::cudaKinematic<TYPE>::
cudaKinematic(
            size_t  Nas, size_t Ndd, size_t Nmesh, double dsp,
            size_t Nt, size_t Npt, double dt,
            const TYPE * const gt0s,
            size_t samples, size_t parameters, size_t observations,
            const size_t * const gidxMap) :
            _Nas(Nas), _Ndd(Ndd), _Nmesh(Nmesh), _dsp(dsp),
            _Nt(Nt), _Npt(Npt), _dt(dt),
            _gt0s(gt0s),
            _samples(samples), _parameters(parameters), _observations(observations),
            _gidx_map(gidxMap)
{
    // create a cublas handle for Gb x Mb
    //cublasSafeCall(cublasCreate(&_cublas_handle));
    // local work sizes
    _Npatch = _Nas*_Ndd;
    _Nddf = (_Ndd+2)*_Nmesh;
    _Nasf = (_Nas+2)*_Nmesh;
    _NGbparameters = 2*_Npatch*_Nt;

    // create work arrays
    initialize(samples);
    // all done
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// initialize the model specific GPU data
/////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename TYPE>
void
altar::models::seismic::cudaKinematic<TYPE>::
initialize(const size_t samples)
{
    // setup work data
    // Mb[samples][Nt][2(strike,dip slips)][Nas][Ndd] leading dimension on right
    cudaSafeCall(cudaMalloc((void**)&_gpu_Mb, (_NGbparameters)*samples*sizeof(TYPE)));
    // gT0 [samples][(Nas+2)*Nmesh][(Ndd+2)*Nmesh] leading dimension on right
    cudaSafeCall(cudaMalloc((void**)&_gpu_T0, _Nddf*_Nasf*samples*sizeof(TYPE)));
    // gTI0[samples][Nas][Npt][Ndd][Npt]
    cudaSafeCall(cudaMalloc((void**)&_gpu_TI0, (_Npatch*_Npt*_Npt)*samples*sizeof(TYPE)));
    // all done
}

// destructor
template <typename TYPE>
altar::models::seismic::cudaKinematic<TYPE>::
~cudaKinematic()
{
    // deallocate GPU
    cudaSafeCall(cudaFree((void*)_gpu_Mb));
    if (_gpu_T0_snapshots) {
        cudaSafeCall(cudaFree((void*)_gpu_T0_snapshots));
        cudaSafeCall(cudaFree((void*)_gpu_dT0));
        cudaSafeCall(cudaFree((void*)_gpu_dTI0));
    }
    cudaSafeCall(cudaFree((void*)_gpu_T0));
    cudaSafeCall(cudaFree((void*)_gpu_TI0));

    //cublasSafeCall(cublasDestroy(_cublas_handle));
    // all don
}


/// @par Main functionality
/// wrap the cudaInitT0 function by a C++ interface for the kinematic model
/// @par CUDA threads layout
///- the total number of threads are the total number of (expanded) mesh points of all the samples that are used for fast sweeping;<br> the number of samples are the leading dimension in the CUDA thread layout
///- one thread corresponds to one (expanded) mesh point of one sample that is used for fast sweeping
///- the number of threads per block is BLOCKDIM*4 (defined in @c altar/utils/common.h);<br> a large block dimension is used to allow more blocks to be lauched for some large systems;<br> if there is still some CUDA launch failure, user can increase it up to 1024 on Tesla M2070/2090
/// @note see @c cudaKinematic_kernels.cu for detailed parameter description
template <typename TYPE>
void
altar::models::seismic::cudaKinematic<TYPE>::
_initT0(const TYPE *const gM, const size_t Nparam, const size_t Ns_good, cudaStream_t stream) const
{
    // set the CUDA block dimenstions
    // use Ns_good (number of samples) as block.z index
    // each xy block(s) treats Nddf x Nasf mesh grids for one sample
    dim3 dim_block(1, BDIMX, BDIMY);
    dim3 dim_grid(Ns_good, IDIVUP(_Nddf, dim_block.y), IDIVUP(_Nasf, dim_block.z));
    /// @note: BLOCKDIM is increased here to accommodate more threads
    cudaKinematic_kernels::initT0_batched<TYPE><<<dim_grid, dim_block, 0, stream>>>(_gidx_map,
        gM, _gpu_T0, Nparam, _Nas, _Ndd, _Nmesh, _dsp, _seed_radius, _seed_steepness, _it0);
    cudaSafeCall(cudaGetLastError());

    /*
    TYPE * hT0 = (TYPE *)malloc(_Nddf*_Nasf*Ns_good*sizeof(TYPE));
    cudaMemcpy(hT0, _gpu_T0, _Nddf*_Nasf*Ns_good*sizeof(TYPE), cudaMemcpyDeviceToHost);
    for(int i=0; i< _Nasf; ++i)
    {
        for(int j =0; j< _Nddf; ++j)
           std::cout << hT0[i*_Nddf+j] << " ";
        std::cout << "\n";
    }
    free(hT0);
    */
}

/// @par Main functionality
/// wrap the cudaSetT0 function by a C++ interface for the kinematic model
/// @par CUDA threads layout
///- the total number of threads are the total number of good samples (pass the "verify" function test)
///- one thread corresponds to one good sample
///- the number of threads per block is BLOCKDIM (defined in @c altar/utils/common.h)
/// @note see @c cudaKinematic_kernels.cu for detailed parameter description
// the threads of a block of the sweeps: the next power of two of the mesh, at least 64
template <typename TYPE>
int
altar::models::seismic::cudaKinematic<TYPE>::
_sweepBlock() const
{
    int meshsize = std::max((_Nas+2)*_Nmesh, (_Ndd+2)*_Nmesh);
    if (meshsize > 1024) fprintf(stderr, "Current fastsweeping cannot support mesh grids large than 1024\n");
    int blockSize = 64;
    while (blockSize < meshsize && blockSize < 1024) blockSize *= 2;
    return blockSize;
}

template <typename TYPE>
void
altar::models::seismic::cudaKinematic<TYPE>::
_fastSweeping(const TYPE *const gM, const size_t Nparam, const size_t Ns_good, cudaStream_t stream,
    TYPE *const gSnapshots) const
{
    // set the CUDA block dimenstions

    dim3 dim_grid(Ns_good), dim_block(_sweepBlock());

    TYPE dspf = _dsp/_Nmesh;
    cudaKinematic_kernels::fastSweeping_batched<TYPE><<<dim_grid, dim_block, 0, stream>>>
        (_gidx_map, gM, _gpu_T0, Nparam, Ns_good, _Nas, _Ndd, _Nmesh, dspf, _sweep_iter, gSnapshots);
    cudaSafeCall(cudaGetLastError());
    /*
        TYPE * hT0 = (TYPE *)malloc(_Nddf*_Nasf*Ns_good*sizeof(TYPE));
    cudaMemcpy(hT0, _gpu_T0, _Nddf*_Nasf*Ns_good*sizeof(TYPE), cudaMemcpyDeviceToHost);
    for(int i=0; i< _Nasf; ++i)
    {
        for(int j =0; j< _Nddf; ++j)
           std::cout << hT0[i*_Nddf+j] << " ";
        std::cout << "\n";
    }
    free(hT0);
    */
}

/// @par Main functionali// wrap the cudaInterpolateT0 function by a C++ interface for the ginematic data part
/// wrap the cudaInterpolateT0 function by a C++ interface for the kinematic model
/// @par CUDA threads layout
///- the total number of threads are the total number of good samples (pass the "verify" function test)
///- one thread corresponds to one good sample
///- the number of threads per block is BLOCKDIM (defined in @c altar/utils/common.h)
/// @note see @c cudaKinematic_kernels.cu for detailed parameter description
template <typename TYPE>
void
altar::models::seismic::cudaKinematic<TYPE>::
_interpolateT0(const size_t Ns_good, cudaStream_t stream) const
{
    // set the CUDA block dimenstions
    dim3 dim_grid(IDIVUP(Ns_good, BLOCKDIM)), dim_block(BLOCKDIM);
    cudaKinematic_kernels::interpolateT0_batched<TYPE><<<dim_grid, dim_block, 0, stream>>>
        (_gpu_T0,  _gpu_TI0, Ns_good, _Nas, _Ndd, _Nmesh, _Npt);
    cudaSafeCall(cudaGetLastError());
}

/// @par Main functionali// wrap the cudaInterpolateT0 function by a C++ interface for the ginematic data part
/// wrap the cudaCastBigM function by a C++ interface for the kinematic model
/// @par CUDA threads layout
///- the total number of threads are the number of samples times the number of patches;<br> the number of samples are the leading dimension in the CUDA thread layout
///- one thread corresponds to one patche of one sample
///- the number of threads per block is BLOCKDIM (defined in @c altar/utils/common.h)
/// @note see @c cudaKinematic_kernels.cu for detailed parameter description
template <typename TYPE>
void
altar::models::seismic::cudaKinematic<TYPE>::
_castBigM(const TYPE *const gM, TYPE *const gMb, const size_t Nparam, const size_t Ns_good, cudaStream_t stream) const
{
    // set the CUDA block dimenstions
    dim3 dim_block(1, BLOCKDIM, 1);
    dim3 dim_grid(Ns_good, IDIVUP(_Npatch, dim_block.y), 1);
    cudaKinematic_kernels::castBigM_batched<TYPE><<<dim_grid, dim_block, 0, stream>>>
        (_gidx_map, gM,  _gpu_TI0, gMb,
            _gt0s, _dt, Nparam, _Nt, _Nas, _Ndd, _Npt);
    cudaSafeCall(cudaGetLastError());
}

template<>
void
altar::models::seismic::cudaKinematic<float>::
linearBigGM(cublasHandle_t handle, const float *const gGb, const float * const gMb, float *gDataPrediction,
        const size_t Ns_good, bool return_residual, cudaStream_t stream) const
{
    // if needed, set the stream to cublas
    // cublasSafeCall(cublasSetStream(_cublas_handle, stream));

    float alpha=1.0f;
    float beta = (return_residual) ? -1.0f : 0.0f;
    // in column-major (c/python)
    //     gRes/gObs(samplesxobs) gMb(samples, NGbparam) gGb(NGbparam, obs)
    // translated to row-major ()
    //     gRes/gObs(obsxsamples) gMb (NGbparam, samples), gGb(obs, NGbparam)
    // therefore, we use gGb x gMb
    int obs = _observations;
    cublasSafeCall(cublasSgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        _observations, Ns_good, _NGbparameters,
        &alpha,
        gGb, obs,
        gMb, _NGbparameters,
        &beta,
        gDataPrediction, _observations));
    // all done
}


template<>
void
altar::models::seismic::cudaKinematic<double>::
linearBigGM(cublasHandle_t handle, const double *const gGb, const double *const gMb, double * const gDataPrediction,
        const size_t Ns_good, bool return_residual, cudaStream_t stream) const
{
    // if needed, set the stream to cublas
    // cublasSafeCall(cublasSetStream(_cublas_handle, stream));

    double alpha=1.0f;
    double beta = (return_residual) ? -1.0f : 0.0f;
    // in column-major (c/python)
    //     gRes/gObs(samplesxobs) gMb(samples, NGbparam) gGb(NGbparam, obs)
    // translated to row-major ()
    //     gRes/gObs(obsxsamples) gMb (NGbparam, samples), gGb(obs, NGbparam)
    // therefore, we use gGb x gMb
    cublasSafeCall(cublasDgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        _observations, Ns_good, _NGbparameters,
        &alpha,
        gGb, _observations,
        gMb, _NGbparameters,
        &beta,
        gDataPrediction, _observations));
    // all done
    // all done
}

// explicit instantiation
template class altar::models::seismic::cudaKinematic<float>;
template class altar::models::seismic::cudaKinematic<double>;

// if having troubles compiling instantiations to shared library
#include "cudaKinematic_kernels.cu"


// end of file
