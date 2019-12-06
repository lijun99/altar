// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-2019 parasim inc
// (c) 2010-2019 california institute of technology
// all rights reserved
//
// Author(s): Hailiang Zhang, Lijun Zhu

// for the build system
#include <portinfo>

// get my class declaration
#include "cudaKinematicG.h"
#include "cudaKinematicG_kernels.h"

// my dependencies
#include <pyre/cuda.h>
#include <algorithm>
#include <iostream>

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// calculate the forward model
// input theta/M (samples x parameters) ld parameters
// construct a Mb (samples x )
/////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename TYPE>
void
altar::models::seismic::cudaKinematicG<TYPE>::
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
altar::models::seismic::cudaKinematicG<TYPE>::
calculateBigM(const TYPE * const theta, TYPE *const gMb, const size_t parameters,
    const size_t batch, cudaStream_t stream) const
{
    // ... and number of good samples
    int good_samples = batch;
    const TYPE * const gM_candidate_queued = theta;

    // set distance/time for 2x2 mesh grids around hypocenter
    _initT0(gM_candidate_queued, parameters, good_samples, stream);
    // find 4 nearest mesh grids close to hypocenter, set their arrival time
    _setT0(gM_candidate_queued, parameters, good_samples, stream);
    // set arrival times for all mesh grids
    _fastSweeping(gM_candidate_queued, parameters, good_samples, stream); // where idx_map comes to play
    // set arrival time for patches (average over mesh grids, but fine-tuned on time intervals Npt)
    _interpolateT0(good_samples, stream);
    // cast to time dependent slips for patches; gMb[samples][Nt][2(strike,dip slips)][Nas][Ndd]
    _castBigM(gM_candidate_queued, gMb, parameters, good_samples, stream); // where idx_map comes to play
    // all done
}

// constructor
template <typename TYPE>
altar::models::seismic::cudaKinematicG<TYPE>::
cudaKinematicG(
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
altar::models::seismic::cudaKinematicG<TYPE>::
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
altar::models::seismic::cudaKinematicG<TYPE>::
~cudaKinematicG()
{
    // deallocate GPU
    cudaSafeCall(cudaFree((void*)_gpu_Mb));
    cudaSafeCall(cudaFree((void*)_gpu_T0));
    cudaSafeCall(cudaFree((void*)_gpu_TI0));

    //cublasSafeCall(cublasDestroy(_cublas_handle));
    // all don
}


/// @par Main functionality
/// wrap the cudaInitT0 function by a C++ interface for the kinematicG model
/// @par CUDA threads layout
///- the total number of threads are the total number of (expanded) mesh points of all the samples that are used for fast sweeping;<br> the number of samples are the leading dimension in the CUDA thread layout
///- one thread corresponds to one (expanded) mesh point of one sample that is used for fast sweeping
///- the number of threads per block is BLOCKDIM*4 (defined in @c altar/utils/common.h);<br> a large block dimension is used to allow more blocks to be lauched for some large systems;<br> if there is still some CUDA launch failure, user can increase it up to 1024 on Tesla M2070/2090
/// @note see @c cudaKernels_KinematicG.cu for detailed parameter description
template <typename TYPE>
void
altar::models::seismic::cudaKinematicG<TYPE>::
_initT0(const TYPE *const gM, const size_t Nparam, const size_t Ns_good, cudaStream_t stream) const
{
    // set the CUDA block dimenstions
    // use Ns_good (number of samples) as block.z index
    // each xy block(s) treats Nddf x Nasf mesh grids for one sample
    dim3 dim_block(BDIMX, BDIMY, 1);
    dim3 dim_grid(IDIVUP(_Nddf, dim_block.x), IDIVUP(_Nasf, dim_block.y), Ns_good);
    /// @note: BLOCKDIM is increased here to accommodate more threads
    cudaKinematicG_kernels::initT0_batched<TYPE><<<dim_grid, dim_block, 0, stream>>>(_gidx_map,
        gM, _gpu_T0, Nparam, _Nas, _Ndd, _Nmesh, _dsp, _it0);
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
/// wrap the cudaSetT0 function by a C++ interface for the kinematicG model
/// @par CUDA threads layout
///- the total number of threads are the total number of good samples (pass the "verify" function test)
///- one thread corresponds to one good sample
///- the number of threads per block is BLOCKDIM (defined in @c altar/utils/common.h)
/// @note see @c cudaKernels_KinematicG.cu for detailed parameter description
template <typename TYPE>
void
altar::models::seismic::cudaKinematicG<TYPE>::
_setT0(const TYPE *const gM, const size_t Nparam, const size_t Ns_good, cudaStream_t stream) const
{
    // set the CUDA block dimenstions
    dim3 dim_grid(IDIVUP(Ns_good, BLOCKDIM)), dim_block(BLOCKDIM);
    cudaKinematicG_kernels::setT0_batched<TYPE><<<dim_grid, dim_block, 0, stream>>>(
        _gidx_map, gM, _gpu_T0, Nparam, Ns_good, _Nas, _Ndd, _Nmesh, _dsp, _it0);
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
/// wrap the cudaFastSweeping function by a C++ interface for the kinematicG model
/// @par CUDA threads layout
///- the total number of threads are the number of good samples (pass the "verify" function test) times the larger number of the (expanded) mesh points along the two fault dimensions;<br> the later is the leading dimension in the CUDA thread layout
///- one thread corresponds to one (expanded) mesh point of one good sample
///- the number of threads per block is the larger number of the (expanded) mesh points along the two fault dimensions
/// @note see @c cudaKernels_KinematicG.cu for detailed parameter description
template <typename TYPE>
void
altar::models::seismic::cudaKinematicG<TYPE>::
_fastSweeping(const TYPE *const gM, const size_t Nparam, const size_t Ns_good, cudaStream_t stream) const
{
    // set the CUDA block dimenstions

    int meshsize = std::max((_Nas+2)*_Nmesh, (_Ndd+2)*_Nmesh);
    int blockSize;
    if (meshsize >1024) fprintf(stderr, "Current fastsweeping cannot support mesh grids large than 1024\n");
    else if(meshsize > 512) blockSize = 1024;
    else if (meshsize > 256) blockSize = 512;
    else if (meshsize > 128) blockSize = 256;
    else if (meshsize > 64) blockSize = 128;
    else blockSize =64;
    dim3 dim_grid(Ns_good), dim_block(blockSize);

    TYPE dspf = _dsp/_Nmesh;
    cudaKinematicG_kernels::fastSweeping_batched<TYPE><<<dim_grid, dim_block, 0, stream>>>
        (_gidx_map, gM, _gpu_T0, Nparam, Ns_good, _Nas, _Ndd, _Nmesh, dspf, _sweep_iter);
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
/// wrap the cudaInterpolateT0 function by a C++ interface for the kinematicG model
/// @par CUDA threads layout
///- the total number of threads are the total number of good samples (pass the "verify" function test)
///- one thread corresponds to one good sample
///- the number of threads per block is BLOCKDIM (defined in @c altar/utils/common.h)
/// @note see @c cudaKernels_KinematicG.cu for detailed parameter description
template <typename TYPE>
void
altar::models::seismic::cudaKinematicG<TYPE>::
_interpolateT0(const size_t Ns_good, cudaStream_t stream) const
{
    // set the CUDA block dimenstions
    dim3 dim_grid(IDIVUP(Ns_good, BLOCKDIM)), dim_block(BLOCKDIM);
    cudaKinematicG_kernels::interpolateT0_batched<TYPE><<<dim_grid, dim_block, 0, stream>>>
        (_gpu_T0,  _gpu_TI0, Ns_good, _Nas, _Ndd, _Nmesh, _Npt);
    cudaSafeCall(cudaGetLastError());
}

/// @par Main functionali// wrap the cudaInterpolateT0 function by a C++ interface for the ginematic data part
/// wrap the cudaCastBigM function by a C++ interface for the kinematicG model
/// @par CUDA threads layout
///- the total number of threads are the number of samples times the number of patches;<br> the number of samples are the leading dimension in the CUDA thread layout
///- one thread corresponds to one patche of one sample
///- the number of threads per block is BLOCKDIM (defined in @c altar/utils/common.h)
/// @note see @c cudaKernels_KinematicG.cu for detailed parameter description
template <typename TYPE>
void
altar::models::seismic::cudaKinematicG<TYPE>::
_castBigM(const TYPE *const gM, TYPE *const gMb, const size_t Nparam, const size_t Ns_good, cudaStream_t stream) const
{
    // set the CUDA block dimenstions
    dim3 dim_block(BLOCKDIM, 1, 1);
    dim3 dim_grid(IDIVUP(_Npatch, dim_block.x), 1, Ns_good);
    cudaKinematicG_kernels::castBigM_batched<TYPE><<<dim_grid, dim_block, 0, stream>>>
        (_gidx_map, gM,  _gpu_TI0, gMb,
            _gt0s, _dt, Nparam, _Nt, _Nas, _Ndd, _Npt);
    cudaSafeCall(cudaGetLastError());
}

template<>
void
altar::models::seismic::cudaKinematicG<float>::
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
altar::models::seismic::cudaKinematicG<double>::
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
template class altar::models::seismic::cudaKinematicG<float>;
template class altar::models::seismic::cudaKinematicG<double>;

// if having troubles compiling instantiations to shared library
#include "cudaKinematicG_kernels.cu"


// end of file
