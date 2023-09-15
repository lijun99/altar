// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2022 - 2023 california institute of technology
// all rights reserved

/**
 * spinup_controller.cuh
 * convergence controller - in a spin up procedure, check whether the convergence is reached
 * it uses the y(tn) values to check convergence
 **/

// code guard
#ifndef cuda_ode_dopri5_spinup_controller_cuh
#define cuda_ode_dopri5_spinup_controller_cuh

#include "external.h"

namespace cuda::ode::dopri5 {

// device object to keep track one system
template<class T>
struct __ALIGNED__ SpinupController
{
    // member variables
    int system_size;
    T atol; // absolute error tolerance
    T rtol;  // relative error tolerance
    T * yold; // keep a copy of previous result

    // methods
    // initialize device, one system per thread
    __device__ void init (const int system_size_, const T atol_, const T rtol_, T * y_)
    {
        atol = atol_;
        rtol = rtol_;
        yold = y_;
        system_size = system_size_;
    };

    // keep copies of y from [i_start, i_end]
    __device__ void record(const cg::thread_block & cta, const T* y, const int i_start, const int i_end)
    {
        auto y_check = y + i_start;
        cuda::detail::vector_copy<T>(cta, yold, y_check, i_end-i_start+1);
    };
    // check convergence
    __device__ bool check_convergence(const cg::thread_block & cta, const T* ynew, const int i_start, const int i_end)
    {
        __shared__ bool converge;
        // define the error estimate function
        auto ynew_check = ynew + i_start;
        auto lambda = [=] (const int i)
        {
            auto val = abs(ynew_check[i]-yold[i])/(atol + rtol*max(abs(ynew[i]), abs(yold[i])));
            return val*val;
        };
        // sum reduction
        auto check_size = i_end - i_start + 1;
        auto val = cuda::detail::sum_block<T, decltype(lambda)>(cta, check_size, lambda);

        if(cta.thread_rank() == 0)
        {
            auto err = sqrt(val/check_size);
            converge = (err <= static_cast<T>(1.0));
            // printf("inside spinup controller err converge %g %d\n", err, converge);
        }
        cta.sync();
        return converge;
    };
        // check convergence, use max formula
    __device__ bool check_convergence2(const cg::thread_block & cta, const T* ynew, const int i_start, const int i_end)
    {
        __shared__ bool converge;
        // define the error estimate function
        auto ynew_check = ynew + i_start;
        auto lambda = [=] (const int i)
        {
            auto val = abs(ynew_check[i]-yold[i])/(atol + rtol*max(abs(ynew_check[i]), abs(yold[i])));
            return val;
        };
        // sum reduction
        auto check_size = i_end - i_start + 1;
        auto err = cuda::detail::max_block<T, decltype(lambda)>(cta, check_size, lambda);

        if(cta.thread_rank() == 0)
        {
            converge = (err <= static_cast<T>(1.0));
            // printf("inside spinup controller err converge %g %d\n", err, converge);
        }
        cta.sync();
        return converge;
    };

};

template <class T>
__global__ void spinup_controller_init_kernel(
    const int systems_batch, const int system_size,
    const T atol, const T rtol,
    T* yolds, SpinupController<T>* controllers)
{
    // get the thread id as system id
    auto system = threadIdx.x + blockIdx.x*blockDim.x;
    if(system < systems_batch)
    {
        // initialize controller for each system
        auto yold = yolds + system*system_size;
        auto & controller = controllers[system];
        controller.init(system_size, atol, rtol, yold);
    }
}

template <class T>
struct SpinupControllerHolder {
    T atol;
    T rtol;
    int systems_batch;
    int system_size;
    T* yolds;
    SpinupController<T> * controllers;

    SpinupControllerHolder(const int systems_batch_, const int system_size_,
        const T atol_, const T rtol_)
        : systems_batch(systems_batch_), system_size(system_size_), atol(atol_), rtol(rtol_)
    {
        // allocate data to save copies of old y
        cudaSafeCall(cudaMalloc(&yolds, systems_batch*system_size*sizeof(T)));
        // allocate device controllers for each system
        cudaSafeCall(cudaMallocManaged(&controllers, systems_batch*sizeof(Controller<T>)));
        // initialize each device controller
        int threads = 256;
        int blocks = (systems_batch-1+threads)/threads; // idivup
        spinup_controller_init_kernel<T><<<blocks, threads>>>(systems_batch, system_size, atol, rtol, yolds, controllers);
        cudaCheckError("spinup_controller_init_kernel error");
    };
    ~SpinupControllerHolder() noexcept(false)
    {
        if(controllers != nullptr)
            cudaSafeCall(cudaFree(controllers));
        if(yolds != nullptr)
            cudaSafeCall(cudaFree(yolds));
    };
};

} // end of namespace cuda::ode::dopri5

#endif // cuda_ode_dopri5_spinup_controller_cuh
// end of file
