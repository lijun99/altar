// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2022 california institute of technology
// all rights reserved
//

// code guard
#if !defined(altar_models_seas_cuda_LinearViscous_h)
#define altar_models_seas_cuda_LinearViscous_h

// declaration
namespace altar::models::seas::cuda {
    namespace linearviscous {

    // the ode solver for linear viscous problem
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
        const T* alpha1 // viscous coefficient
        );

    } // end of namespace linearviscous
} // end of namespace altar::models::seas::cuda
#endif
