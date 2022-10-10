// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2022 california institute of technology
// all rights reserved
//

#ifndef __linearviscous_ode_cuh__
#define __linearviscous_ode_cuh__

// my dependencies
#include "dopri5.cuh"
#include <pyre/cuda.h>

// global namespace enclosure
namespace altar::models::seas::cuda {

// linear viscous methods
namespace linearviscous_ode {

// for each model, defined
// 1. ode_function - define the ode function
// 2. ode_solver_kernel - cuda kernel for a specific ode function, needed because each sample may have different varied parameters
// 3. struct/class to wrap data and run method, served as python interface

// 1. function used in the ode integration
// the first four parameters are required as standard ones in an ode equation
// the rest may be defined by the model
// here y = (slip, velocity)  [2*patches] for creeping zone
// return f = dy/dt = (ds/dt, dv/dt)
template <typename T>
struct ode_function {
    __device__ __host__ void operator()
        (T* f, //  dydt output vector [2*patches]
        const T t, // time
        const T* y, // y value, vector [2*patches]
        const int system_size, // 2*patches
        const T Vj, // backslip rate
        const T* stressKernel, // stress kernel matrix [patches,patches]
        const T* stressrate_ext, // stress rate imposed by external patches
        const int parameters, const T *alpha1 // viscous coefficient
        )
   {
        // the system_size is 2*patches, slip and velocity
        int patches = system_size/2;
        // get the physical quantities from wrapped data
        auto dsdt = f;
        auto dvdt = f+patches;
        // auto slip = y; // not used
        auto velocity = y+patches;

        // set dsdt
        for(int i=0; i<patches; ++i)
            dsdt[i] = velocity[i] - Vj;

        // set dvdt
        for(int ix=0; ix<patches; ++ix) {
            // use dvdt for dtau/dt temporarily)
            dvdt[ix] = stressrate_ext[ix];
            for(int iy=0; iy<patches; ++iy)
                dvdt[ix] += (velocity[iy]-Vj) *stressKernel[iy*patches+ix];
            // get dvdt from dtau/dt
            dvdt[ix] = dvdt[ix]/alpha1[0];
        }
        // all done return f
   }
}; // end of struct ode_function

// 2. cuda kernels to call rk sovler to solve a batch of samples
// one thread for each sample
// need to assign varied parameters to each sample
template <typename T>
__global__ void ode_solver_kernel(
    const int rk_steps,
    T* rk_work,
    const int samples,
    const int system_size,
    const T t0, const T t1,
    const T* y0,
    bool is_dense_output,
    const T* t_out, T* y_out, const int n_out,
    ode_function<T> dydt,
    const T Vj, const T* stressKernel, const T* stressrate_ext, // model-depend parameters
    const int parameters, const T* alpha1
    )
{
    // one thread per sample, to get the sample index
    int sample = blockIdx.x *blockDim.x + threadIdx.x;

    // check thread id in range of samples
    // - a common cuda check since threads could be larger than samples
    if(sample >= samples)
        return;
    // get the starting pointer for samples
    auto y0_s = y0 + sample*system_size; // initial values of (slip, velocity)
    auto yout_s = y_out + sample*system_size*n_out;
    // get the varies parameter for this sample
    auto alpha1_s = alpha1 + sample*parameters;
    // get the work data for each sample 14*system_size
    auto rk_work_s = rk_work + sample*14*system_size;

    // first bracket <...>:
    // T-typename, ode_function<T>-function type
    // const T, const T, const T* - model-depend parameter types
    ode::dopri5::rk_solver_fixedstep(
        rk_steps, rk_work_s,
        system_size,
        t0, t1,
        y0_s,
        is_dense_output,
        t_out, yout_s, n_out,
        dydt,
        Vj, stressKernel, stressrate_ext,
        parameters, alpha1_s
        );
    // printf("ode kernel %d %d %g %g\n", sample, samples, yout_s[0], yout_s[system_size*n_out-1]);
    // all done
}


// 3. define the struct/class ode_solver to wrap data and methods
// see LinearViscous.h for defintion

template <typename T>
void ode_solver(
    const int rk_steps, // number of rk steps between t0 and tn
    const int samples,
    const int system_size, //system_size = 2*patches for slip and velocity
    const T t0, // start time
    const T tn, // end time
    const T* y0, // initial values for y =(slip, velocity) [samples, 2*patches]
    const bool dense_output,
    const T* tout, // desired output time points [samples, nout]
    T *yout, // output y values at tout  [samples, nout, 2*patches]
    const int nout, // number of desired output time points
    const T Vj,
    const T* stressKernel,
    const T* stressrate_ext,
    const int parameters,
    const T* alpha1 // viscous coefficient, a constant for all patches in each sample [samples]
    )
{

    // create an instance of ode_function
    ode_function<T> dydt;

    // compute the number of gpu blocks needed
    const int threadsPerBlock = 256;
    const int numberOfBlocks = (samples-1+threadsPerBlock)/threadsPerBlock; //IDIVUP

    // allocate the work data for rk ode solver, will move this to a better location of the code
    T * rk_work;
    // 9 vectors for rk solver, 5 vectors for dense output
    cudaSafeCall(cudaMalloc(&rk_work, 14*samples*system_size*sizeof(T)));

    // call ode solver kernel
    ode_solver_kernel<<<numberOfBlocks, threadsPerBlock>>>(
        rk_steps, rk_work,
        samples,
        system_size, // system size
        t0, tn,
        y0,
        dense_output, // dense out = true
        tout, yout, nout, // for dense_output
        dydt, // the above are stand parameters to call rk_solver
        Vj, stressKernel, stressrate_ext,  // T* T T*
        parameters, alpha1 // int T*
        );
    // check errors
    cudaSafeCall(cudaGetLastError());
    // free the work data
    cudaSafeCall(cudaFree(rk_work));
    // all done
}

// explicit instantiation for python module
template void ode_solver<double>(const int, const int,  const int, const double, const double,
    const double*, const bool, const double*, double*, const int, const double, const double*, const double*,
    const int, const double*);
template void ode_solver<float>(const int, const int,  const int, const float, const float,
    const float*, const bool, const float*, float*, const int, const float, const float*, const float*,
    const int, const float*);

} // namespace linearviscous_ode
} // namespace

#endif
// end of file
