// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2022-2023 california institute of technology
// all rights reserved

/**
 * odesystem.cuh
 * This is a template of ode system
 * when following the template, please keep the naming convention for member variables and functions
 **/

// code guard
#ifndef cuda_ode_dopri5_odesystem_cuh
#define cuda_ode_dopri5_odesystem_cuh

#include "external.h"

namespace cuda::ode::dopri5 {

template <class T>
struct __ALIGNED__ OdeSystem {
    // each system is defined as #patches and #units in a patch
    int patches; // patches per system, one patch processed by one GPU thread
    int units; // units per batch, e.g., number of degrees of freedom for one patch/grid
    int system_size; // patches*units
    int systems; // total systems to be processed in a batch

    // add other shared variables here
    // e.g. T * theta

    // constructor
   OdeSystem(const int p, const int u, const int sys)
        : patches(p), units(u), systems(sys), system_size(p*u)
    {
    };

    // ode function for a given patch f= dy/dt = f(t, y)
    __device__ void dydt(const int system_id, const int patch_id, const T t, const T* y, T* f)
    {
        // default, set to 0
        for (auto unit_id =0; unit_id < units; unit_id++)
            f[patch_id+unit_id*patches] = static_cast<T>(0);
        return;
    };

    // ode function called when solving a system with a thread block
    __device__ void dydt_block(const cg::thread_block & cta,
        const int system_id, const T t, const T* y0, T* f)
    {
        for(int patch_id = cta.thread_rank(); patch_id<patches; patch_id+=cta.size())
            dydt(system_id, patch_id, t, y0, f);
    };

    // also add other methods, e.g.,
    // void init_parameters(args...); to initialize shared variables

};

} // end of namespace cuda::ode::dopri5

#endif // cuda_ode_dopri5_odesystem_cuh
// end of file
