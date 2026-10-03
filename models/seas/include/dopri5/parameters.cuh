// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2022 - 2023 california institute of technology
// all rights reserved

/**
 * parameters.cuh
 * fixed controlling parameters reside in gpu constant memory
 **/

// code guard
#ifndef cuda_ode_dopri5_parameters_cuh
#define cuda_ode_dopri5_parameters_cuh

// external dependencies
#include "external.h"

namespace cuda::ode::dopri5 {

    // use a namespace to enclose the parameters in order to differentiate from local variables
    namespace parameters {

        // number of variables in ode
        __device__ __constant__ int system_size;

        // number of systems in a batch
        __device__ __constant__ int systems_in_a_batch;

        // absolute error in ODE stepper
        template <typename T>
        __device__ __constant__ T atol;

        // relative error in ODE stepper
        template <typename T>
        __device__ __constant__ T rtol;


        // constant parameters for adaptive step size
        template <typename T>
        __device__ __constant__ T beta;

        template <typename T>
        __device__ __constant__ T alpha;

        template <typename T>
        __device__ __constant__ T minscale;

        template <typename T>
        __device__ __constant__ T maxscale;

        template <typename T>
        __device__ __constant__ T safety;


        void set_system_size (const int size)
        {
            // copy parameters to constant memory in gpu
            cudaSafeCall(cudaMemcpyToSymbol(system_size, &size, sizeof(int)));
        }

        void set_systems_in_a_batch(const int size)
        {
            cudaSafeCall(cudaMemcpyToSymbol(systems_in_a_batch, &size, sizeof(int)));
        }

        template <typename T>
        void set_tolerance (const T atol_, const T rtol_)
        {
            // copy parameters to constant memory in gpu
            cudaSafeCall(cudaMemcpyToSymbol(atol<T>, &atol_, sizeof(T)));
            cudaSafeCall(cudaMemcpyToSymbol(rtol<T>, &rtol_, sizeof(T)));
        }

        // init rk54 adaptive step size parameters
        // h_new = safety * h_current (tolerance/error estimate)^alpha
        // h_new = min(h_new, maxscale*h_current)
        // h_new = max(h_new, minscale*h_current)
        template <typename T>
        void set_rk45_adaptive_parameters()
        {

            // adjust beta if using
            auto beta_ = static_cast<T>(0.0); // 0.4/k
            auto alpha_ = static_cast<T>(0.2)
                -beta_*static_cast<T>(0.75); // 1/k - 0.75\beta
            auto minscale_ = static_cast<T>(0.2);
            auto maxscale_ = static_cast<T>(10.0);
            auto safety_ = static_cast<T>(0.9);

            // copy parameters to constant memory in gpu
            cudaSafeCall(cudaMemcpyToSymbol(beta<T>, &beta_, sizeof(T)));
            cudaSafeCall(cudaMemcpyToSymbol(alpha<T>, &alpha_, sizeof(T)));
            cudaSafeCall(cudaMemcpyToSymbol(minscale<T>, &minscale_, sizeof(T)));
            cudaSafeCall(cudaMemcpyToSymbol(maxscale<T>, &maxscale_, sizeof(T)));
            cudaSafeCall(cudaMemcpyToSymbol(safety<T>, &safety_, sizeof(T)));
        }

        // a common interface to set the parameters
        template <typename T>
        void initialize(const int system_size, const T atol, const T rtol)
        {
            set_system_size (system_size);
            set_tolerance(atol, rtol);
            set_rk45_adaptive_parameters();
        }


    }

    // define a short name for the namespace
    namespace p = parameters;

}


#endif // cuda_ode_dopri5_parameters_cuh
