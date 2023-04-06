// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2022 - 2023 california institute of technology
// all rights reserved

/**
 * Ode.cuh
 * Define the rate-dependent Ode Functions
 **/

#ifndef altar_models_seas_cuda_ratedependent_ode_cuh
#define altar_models_seas_cuda_ratedependent_ode_cuh

#include "math.h"

// enclosed in a name space
namespace altar::models::seas::cuda::ratedependent {

template <class T>
struct Ode {
    // required parameters, keep their names
    // @note each system is defined by #patches and each patch with #units
    // @note each patch is processed by one thread
    // @note here, zeta is the normalized logarithmic velocity with v = Vj * exp(zeta)
    int patches; // number of patches per system
    int units;   // number of units per patch, should be 2, dsdt and dzetadt
    int system_size; // patches*units
    int systems; // total systems/samples to be processed

    // other custom parameters
    // all these parameters need to set inside this structure
    int parameters; // size of theta per system
    const T *alpha_h_vec; // viscous coefficient for each patch [systems, patches]
    T Vj; // backslip rate
    T mu_over_2vs; // radiation damping coefficient
    T* stress_kernel; // stress kernel matrix [patches,patches]
    T* stressrate_ext; // stress rate imposed by external patches

    // ode function for a given patch f = dy/dt = f(t, y), processed by one gpu thread
    // @parameter y [units*patches]: units and patches may be arranged in any prescribed order, per convenience
    //  here, we use [units, patches], i.e., [s0, s1, ...., zeta0, zeta1, ...]
    // @return f, same order as y, i.e., [dsdt0, dsdt1, ..., dzetadt0, dzetadt1, ...]
    __device__ __forceinline__ void dydt(const int system_id, const int patch_id, const T t, const T* y, T* f)
    {

        // get velocity pointer
        auto zeta = y + patches;

        // dsdt = v = Vj * zeta
        auto v = Vj * exp(zeta[patch_id])
        f[patch_id] = v; // dsdt

        // dvdt
        // compute radiation damping
        auto raddamp = mu_over_2vs * v / alpha_h_vec[system_id*patches+patch_id]
        // compute dtau/dt, put it into f
        auto ix = patches + patch_id;
        f[ix] = stressrate_ext[patch_id];
        for (int iy=0; iy<patches; ++iy)
            f[ix] += Vj * (exp(zeta[iy]) - 1) * stress_kernel[iy*patches+ix];
        // apply radiation damping
        f[ix] /= 1 + raddamp
        // get dvdt from dtau/dt
        f[ix] /= alpha_h_vec[system_id*patches+patch_id];
        // all done for this patch
        return;
    };

    // hook called by the ode solver, keep it as it is
    // ode function called when solving a system with a thread block
    __device__ __forceinline__  void dydt_block(const cg::thread_block& cta, const int system_id, const T t, const T* y0, T* f)
    {
        // this loop is needed because the total number of patches may be bigger than the total number of threads
        for (int patch_id = cta.thread_rank(); patch_id<patches; patch_id+=cta.size())
            dydt(system_id, patch_id, t, y0, f);
    };

    // pass the fixed and pre-allocated parameters
    // called by model at initialization
    void init_parameters(
        const T Vj_, const T mu_over_2vs_, T* stress_kernel_, T* stressrate_ext_)
    {
        Vj = Vj_;
        mu_over_2vs = mu_over_2vs_;
        stress_kernel = stress_kernel_;
        stressrate_ext = stressrate_ext_;
    };

    // pass the updated parameters
    // assume theta has shape [systems, parameters]
    void set_alpha_h(const T* theta_, const int parameters_)
    {
        parameters = parameters_;
        T log_alpha_h_range;
        T frac_log_alpha_h;
        for (int isys=0, isys<systems, ++isys)
        {
            log_alpha_h_range = theta_[isys*parameters+1] - theta_[isys*parameters];
            for (int ipatch=0, ipatch<patches, ++ipatch)
            {
                // assume log-linear slope between boundary alpha_h values in theta
                // this still assumes patch order is meaningful and 1D
                frac_log_alpha_h = theta_[isys*parameters] + ipatch * log_alpha_h_range / patches;
                alpha_h_vec[isys*patches+ipatch] = pow(10, frac_log_alpha_h);
            };
        };
    };

    // constructor
    Ode(const int p, const int u, const int sys)
        : patches(p), units(u), systems(sys), system_size(p*u)
    {
    };
};

} // end of namespace

#endif
// end of file
