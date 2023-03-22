// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2022 - 2023 california institute of technology
// all rights reserved

/**
 * Ode.cuh
 * Define the Linear Viscous Ode Functions
 **/

#ifndef altar_models_seas_cuda_linearviscous_ode_cuh
#define altar_models_seas_cuda_linearviscous_ode_cuh

// enclosed in a name space
namespace altar::models::seas::cuda::linearviscous {

template <class T>
struct Ode {
    // required parameters, keep their names
    // @note each system is defined by #patches and each patch with #units
    // @note each patch is processed by one thread
    int patches; // number of patches per system
    int units;   // number of units per patch, should be 2, dsdt and dvdt
    int system_size; // patches*units
    int systems; // total systems/samples to be processed

    // other custom parameters
    // all these parameters need to set inside this structure
    int parameters; // number of alpha1 per system
    const T *alpha1; // viscous coefficient [systems, parameters]
    T Vj; // backslip rate
    T* stress_kernel; // stress kernel matrix [patches,patches]
    T* stressrate_ext; // stress rate imposed by external patches

    // ode function for a given patch f = dy/dt = f(t, y), processed by one gpu thread
    // @parameter y [units*patches]: units and patches may be arranged in any prescribed order, per convenience
    //  here, we use [units, patches], i.e., [s0, s1, ...., v0, v1, ...]
    // @return f, same order as y, i.e., [dsdt0, dsdt1, ..., dvdt0, dvdt1, ...]
    // @note assuming
    __device__ __forceinline__ void dydt(const int system_id, const int patch_id, const T t, const T* y, T* f)
    {

        // get velocity pointer
        auto velocity = y + patches;

        // dsdt = v -Vj
        f[patch_id] = velocity[patch_id] - Vj; // dsdt

        // dvdt
        // compute dtau/dt at first
        auto ix = patches+patch_id;
        f[ix] = stressrate_ext[patch_id];
        for(int iy=0; iy<patches; ++iy)
            f[ix] += (velocity[iy]-Vj) *stress_kernel[iy*patches+ix];
        // get dvdt from dtau/dt
        f[ix] /= alpha1[system_id];
        // all done for this patch
        return;
    };

    // hook called by the ode solver, keep it as it is
    // ode function called when solving a system with a thread block
    __device__ __forceinline__  void dydt_block(const cg::thread_block& cta, const int system_id, const T t, const T* y0, T* f)
    {
        // this loop is needed because the total number of patches may be bigger than the total number of threads
        for(int patch_id = cta.thread_rank(); patch_id<patches; patch_id+=cta.size())
            dydt(system_id, patch_id, t, y0, f);
    };

    // pass the fixed and pre-allocated parameters
    // called by model at initialization
    void init_parameters(
        const T Vj_, T* stress_kernel_, T* stressrate_ext_)
    {
        Vj = Vj_;
        stress_kernel = stress_kernel_;
        stressrate_ext = stressrate_ext_;
    };

    // pass the updated parameters
    void set_alpha1(const T* alpha1_, const int parameters_)
    {
        parameters = parameters_;
        alpha1 = alpha1_;
    }

    // constructor
    Ode(const int p, const int u, const int sys)
        : patches(p), units(u), systems(sys), system_size(p*u)
    {
    };
};

} // end of namespace

#endif
// end of file
