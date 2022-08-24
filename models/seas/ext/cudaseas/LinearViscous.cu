// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2022 california institute of technology
// all rights reserved
//

// for the build system
//#include <portinfo>

// get my class declaration
#include "LinearViscous.h"

// my dependencies
#include "dopri5.cuh"
#include <stdio.h>
#include <iostream>
// #include <pyre/cuda.h>
// #include <algorithm>

// global namespace enclosure
namespace altar::models::seas::cuda {

// linear viscous methods
namespace linearviscous {

// for each model, defined
// 1. ode_function - define the ode function
// 2. ode_solver_kernel - cuda kernel for a specific ode function, needed because each sample may have different varied parameters
// 3. struct/class to wrap data and run method, served as python interface

// 1. function used in the ode integration
// the first four parameters are required as standard ones in an ode equation
// the rest may be defined by the model
// here, y = (s, v)  [2*patches]
template <typename T>
struct ode_function {
    __device__ __host__ void operator()
        (T* f, //  dydt output vector [2*patches]
        const T t, // t0
        const T* y, // slip value, vector [2*patches]
        const int system_size, // 2*number of patches
        const int asperity_range, // the range of creeping zone
        const T alpha1, // viscous coefficient
        const T Vj, // backslip rate
        const T* stressKernel // stress kernel matrix [patches,patches]
        )
   {
        // the system_size is 2*patches, slip and velocity
        auto patches = system_size/2;
        // get the physical quantities from wrapped data
        auto dsdt = f;
        auto dvdt = f+patches;
        // auto slip = y; // not used
        auto velocity = y+patches;

        // set dsdt
        for(int i=0; i<patches; ++i)
            dsdt[i] = velocity[i];

        // set dvdt
        for(int ix=0; ix<patches; ++ix) {
            // use dvdt for dtau/dt temporarily)
            dvdt[ix] = 0;
            if(ix>=asperity_range)
                for(int iy=0; iy<patches; ++iy)
                    dvdt[ix] += velocity[iy]*stressKernel[iy*patches+ix];
            // get dvdt from dtau/dt
            dvdt[ix] = dvdt[ix]/alpha1 - Vj;
        }
        // all done return f
   }
}; // end of struct ode_function

// 2. cuda kernels to call rk sovler to solve a batch of samples
// need to be customized for each model on how to assign varied parameters to each sample
template <typename T>
__global__ void ode_solver_kernel(
    const int samples,
    const int system_size,
    const T t0, const T t1, const int steps,
    const T* y0,
    bool is_dense_output,
    const T* t_out, T* y_out, const int n_out,
    ode_function<T> dydt,
    const int asperity_range, const T* alpha1, const T Vj, const T* stressKernel // model-depend parameters
    )
{
    // one thread per sample, to get the sample index
    int sample = blockIdx.x *blockDim.x + threadIdx.x;

    // check thread id in range of samples
    if(sample >= samples)
        return;
    // get the starting pointer for samples
    auto y0_s = y0 + sample*system_size;
    auto yout_s = y_out + sample*system_size*n_out;

    // get the varies parameter for this sample
    auto alpha1_s = alpha1[sample];

    // first bracket <...>:
    // T-typename, ode_function<T>-function type
    // const T, const T, const T* - model-depend parameter types

    ode::dopri5::rk_solver_fixedstep<T, ode_function<T>, const int, const T, const T, const T*>(
        system_size,
        t0, t1, steps,
        y0_s,
        is_dense_output,
        t_out, yout_s, n_out,
        dydt,
        asperity_range,
        alpha1_s, Vj, stressKernel);
    // all done
}


// 3. define the struct/class ode_solver to wrap data and methods
// see LinearViscous.h for defintion


// 3.2 cuda ode solver
template <typename T>
void ode_solver(
    const int samples,
    const int system_size, //system_size = 2*patches for slip and velocity
    const T t0, // start time
    const T tn, // end time
    const int rk_steps, // number of rk steps between t0 and tn
    const T* y0, // initial values for y =(slip, velocity) [samples, 2*patches]
    const bool dense_output,
    const T* tout, // desired output time points [samples, nout]
    T *yout, // output y values at tout  [samples, nout, 2*patches]
    const int nout, // number of desired output time points
    const int asperity_range, // creep zone range - fixed parameters
    const T Vj,
    const T* stressKernel,
    const T* alpha1 // viscous coefficient, a constant for all patches in each sample [samples]
    )
{

    // create an instance of ode_function
    ode_function<T> func;

    // compute the number of gpu blocks needed
    const int threadsPerBlock = 256;
    const int numberOfBlocks = (samples-1+threadsPerBlock)/threadsPerBlock; //IDIVUP
    std::cout << numberOfBlocks;


    // call ode solver kernel
    ode_solver_kernel<T><<<numberOfBlocks, threadsPerBlock>>>(
        samples,
        system_size, // system size
        t0, tn, rk_steps,
        y0,
        dense_output, // dense out = true
        tout, yout, nout, // for dense_output
        func, // the above are stand parameters to call rk_solver, below are model-depend parameter, included in args...
        asperity_range, // int
        alpha1, Vj, stressKernel  // T* T T*
        );
    // check errors
    auto status = cudaGetLastError();
    if (status != cudaSuccess)
        printf("CUDA Error Code %d: %s - at %s:%d\n",
                status, cudaGetErrorString(status), __FILE__, __LINE__);
    // all done
}

// explicit instantiation for python module
template void ode_solver<double>(const int, const int, const double, const double, const int,
    const double*, const bool, const double*, double*, const int, const int, const double, const double*, const double*);
template void ode_solver<float>(const int, const int, const float, const float, const int,
    const float*, const bool, const float*, float*, const int, const int, const float, const float*, const float*);

} // namespace linearviscous
} // namespace


// end of file
