// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-2019 parasim inc
// (c) 2010-2019 california institute of technology
// all rights reserved
//
// Author(s): Hailiang Zhang, Lijun Zhu

// code guard
#if !defined(altar_models_seismic_cudaKinematicG_h)
#define altar_models_seismic_cudaKinematicG_h



// macros
#include <cublas_v2.h>

// place everything in the local namespace
namespace altar {
    namespace models {
        namespace seismic {
            // forward declarations
            template<typename TYPE> class cudaKinematicG;
        } // of namespace seismic
    } // of namespace models
} // of namespace altar


/// @brief The kinematic model with big-G implementation on GPU
///
/// @par Primary contents
/// the big-G matrix on GPU<br>
/// the big-M matrix on GPU<br>
/// the T0 values for all patches on GPU<br>
///
/// @par Main functionalities
/// data-LLK calculation on GPU<br>
/// wrappers of cuda kernels for fast sweeping and related functions
///
/// @note
/// none
///

// declaration
template <typename TYPE>
class altar::models::seismic::cudaKinematicG
{
    // data
protected:
    size_t _Nas; ///< the number of patches along the strike direction
    size_t _Ndd; ///< the number of patches along the dip direction
    size_t _Nt; ///< the number of small triangle source time functions
    size_t _Npt; ///< the number of source time functions for T0 interpolation (along each dimension; each patch will have _Npt*_Npt source time functions)
    TYPE _dsp; ///< the length of a single patch along a single dimension
    TYPE _dt; ///< the width of the small triangle source time function
    size_t _Nmesh; ///< the number of grid point on each patch for fast-sweeping

    // work matrices/vectors
    TYPE * _gpu_Mb; ///< the big-M matrix ((2*<c>_Nt</c>*Npatch)*Ns where Ns is the leading index) on GPU
    TYPE * _gpu_T0; ///< the T0 values for each patch ( ((Nas+2)*(Ndd+2)*Nmesh*Nmesh) * Ns where Ns is the leading index) on GPU
    TYPE * _gpu_TI0; ///< the T0 values for each interpolated points ((Np*Npt*Npt)*Ns where Ns is the leading index) on GPU
    // with leading dimension on the right
    // Mb[samples][Nt][2(strike,dip slips)][Nas][Ndd]
    // gT0 [samples][(Nas+2)*Nmesh][(Ndd+2)*Nmesh]
    // gTI0[samples][Nas][Npt][Ndd][Npt]
    size_t _Nddf; //(Ndd+2)*Nmesh
    size_t _Nasf; // (Nas+2)*Nmesh
    size_t _Npatch; // Nas*Ndd
    size_t _NGbparameters; //[Nt][2(strike,dip slips)][Nas][Ndd]

    // input
    const size_t * _gidx_map; ///< parameter indices
    // arranged as (strike slips x Npatch, dip slips x Npatch, rupture time x Npatch, rupture velocity x Npatch,
    //   hypocenter along strike, hypocenter along dip)
    // The patches are arranged as [Nas][Ndd] with leading dimension Ndd
    const TYPE * _gt0s; ///< the t0 values of small triangle source time funtion of all patches (Np) on GPU

    // simulation sizes
    size_t _samples;
    size_t _observations;
    size_t _parameters; // number of parameters in gM/theta, not neccesarily this model

    const TYPE _it0 = 1.e6; ///< large arrival time for fastsweep
    const size_t _sweep_iter = 1; ///< number of iterations for fastsweeping

    cublasHandle_t _cublas_handle;

    // method
public:
    /// initialize the model specific GPU data
    void initialize(const size_t samples);
    /// calculate the forward model
    void forwardModel(cublasHandle_t handle, const TYPE * const theta, const TYPE * const Gb, TYPE * const prediction,
        const size_t parameters, const size_t batch, bool return_residual=true, cudaStream_t stream=0) const;
    /// calculate and return the bigM only
    void calculateBigM(const TYPE * const theta, TYPE * const gMb, const size_t parameters, const size_t batch, cudaStream_t stream=0) const;

    // local methods
    /// Initialize the T0 data
    void _initT0(const TYPE *const gM, const size_t Nparam, const size_t Ns_good, cudaStream_t stream=0) const;
    /// Set t0 for the 4 patches closest to hypo center
    void _setT0(const TYPE *const gM, const size_t Nparam, const size_t Ns_good, cudaStream_t stream=0) const;
    /// wrapper for Fast Sweeping kernel
    void _fastSweeping(const TYPE *const gM, const size_t Nparam, const size_t Ns_good, cudaStream_t stream=0) const;
    /// wrapper for interpolation of T0 to TI0 kernel
    void _interpolateT0(const size_t Ns_good, cudaStream_t stream=0) const;
    /// wrapper for casting M_candidate to BigM kernel
    void _castBigM(const TYPE *const gM, TYPE * const gMb, const size_t Nparam, const size_t Ns_good, cudaStream_t stream=0) const;
    /// perform the BigM x BigG
    void linearBigGM(cublasHandle_t handle, const TYPE *const gGb, const TYPE * const gMb,
        TYPE * const gDataPrediction, const size_t Ns_good, bool return_residual, cudaStream_t stream=0) const;

    // meta-methods
public:
    /// constructor
    cudaKinematicG(
            size_t Nas, size_t Ndd, size_t Nmesh, double dsp, //patch info
            size_t Nt, size_t Npt, double dt, // time info
            const TYPE * const gt0s, // starting time
            size_t samples, size_t parameters, size_t observations, // simulation info
            const size_t * const gidxMap);
    /// destructor
    virtual ~cudaKinematicG();

    // disallow
private:
    /// copy constructor disallowed
    inline cudaKinematicG(const cudaKinematicG &);
    /// assign constructor disallowed
    inline const cudaKinematicG & operator=(const cudaKinematicG &);
};

#endif
