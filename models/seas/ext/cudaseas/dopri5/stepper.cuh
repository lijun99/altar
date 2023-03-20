// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2022 - 2023 california institute of technology
// all rights reserved

/**
 * dopri5_stepper.cuh
 * Runge-Kutta Dormand Prince 54 ODE stepper - integrate over (t0, t0+h)
 * Butcher Tableau
 template<typename T>
    struct tableau {
    static const int stage = 6;
    static const int order = 5;
    static const int error_estimated_order = 4;
    // C
    static const T c2=0.2,c3=0.3,c4=0.8,c5=8.0/9.0;
    // A
    static const T a21=0.2,
        a31=3.0/40.0,a32=9.0/40.0,
        a41=44.0/45.0, a42=-56.0/15.0, a43=32.0/9.0,
        a51=19372.0/6561.0, a52=-25360.0/2187.0, a53=64448.0/6561.0, a54=-212.0/729.0,
        a61=9017.0/3168.0, a62=-355.0/33.0, a63=46732.0/5247.0, a64=49.0/176.0, a65=-5103.0/18656.0;
    // B
    static const T b1=35.0/384.0, b3=500.0/1113.0, b4=125.0/192.0, b5=-2187.0/6784.0, b6=11.0/84.0;
    // E (error)
    static const T e1=71.0/57600.0, e3=-71.0/16695.0, e4=71.0/1920.0,
	    e5=-17253.0/339200.0, e6=22.0/525.0, e7=-1.0/40.0;
};
 **/

// code guard
#ifndef cuda_ode_dopri5_stepper_cuh
#define cuda_ode_dopri5_stepper_cuh

#include "external.h"
#include "detail.cuh"

// enclose inside a namespace
namespace cuda::ode::dopri5 {

// Stepper State in
template <class T>
struct __ALIGNED__ Stepper{
    int patches;
    int units;
    int system_size;
    T* y0; // [size] initial value
    T* yn; // [size] final value at t=t0+h
    T* k1; // f(t0, y0)
    T *k2, *k3, *k4, *k5, *k6;
    T* k7; // f(t0+h, yn)
    T *en; // error estimate

    // initialize the pointers
    __device__ void init(const int pps, const int upp, T* rk_work)
    {
        // set size
        patches = pps;
        units = upp;
        system_size = patches*units;
        // set up pointers to the work data
        y0 = rk_work;
        yn = rk_work +    system_size;
        k1 = rk_work +  2*system_size;
        k2 = rk_work +  3*system_size;
        k3 = rk_work +  4*system_size;
        k4 = rk_work +  5*system_size;
        k5 = rk_work +  6*system_size;
        k6 = rk_work +  7*system_size;
        k7 = rk_work +  8*system_size;
        en = rk_work +  9*system_size;
    };

    template <class ode_system_type>
    __device__ void integrate(const cg::thread_block& cta, const int sys_id, const T t0, const T h, ode_system_type& dydt);

    template <class ode_system_type>
    __device__ void set_init_value(const cg::thread_block& cta,
            const T t0, const T* y0_in, const int system_id, ode_system_type& ode)
    {
        // copy y0
        cuda::detail::vector_copy<T>(cta, y0, y0_in, system_size);
        // set f0 = f(t0, y0)
        ode.dydt_block(cta, system_id, t0, y0, k1);
    };

    template <class ode_system_type>
    __device__ void set_f0_value(const cg::thread_block& cta,
            const T t0, const T* y0_in, const int system_id, ode_system_type& ode)
    {
        // set f0 = f(t0, y0)
        ode.dydt_block(cta, system_id, t0, y0, k1);
    };

    template <class ode_system_type>
    __device__ void set_f0_value(const cg::thread_block& cta,
            const T t0, const int system_id, ode_system_type& ode)
    {
        // set f0 = f(t0, y0)
        ode.dydt_block(cta, system_id, t0, y0, k1);
    };

};

template <class T>
struct StepperHolder {
    int patches;
    int units;
    int systems_batch;
    Stepper<T> * steppers;
    T * work;

    // constructor
    StepperHolder (const int pps, const int upp, const int sys);
    // destructor
    ~StepperHolder() noexcept(false);
};


template <class T>
__global__ void stepper_state_init_kernel(const int patches, const int units, const int systems_batch, T* work,
    Stepper<T>* steppers)
{
    // get the thread id as system id
    auto system = threadIdx.x + blockIdx.x*blockDim.x;
    if(system < systems_batch)
    {
        // initialize stepper for each system
        auto & stepper = steppers[system];
        auto work_system = work + 10*patches*units*system;
        stepper.init(patches, units, work_system);
    }
}

template <class T>
StepperHolder<T>::StepperHolder(const int pps, const int upp, const int sys)
    : patches(pps), units(upp), systems_batch(sys)
{
    //allocate the work cache
    cudaSafeCall(cudaMalloc(&work,
        10*patches*units*systems_batch*sizeof(T)));
    // allocate the device StepperHolder
    cudaSafeCall(cudaMalloc(&steppers, systems_batch*sizeof(Stepper<T>)));

    // initialize each stepper
    int threads = 256;
    int blocks = (systems_batch-1+threads)/threads; // idivup
    stepper_state_init_kernel<T><<<blocks, threads>>>(patches, units, systems_batch, work, steppers);
    cudaCheckError("stepper_state_init_kernel error");
}

template <class T>
StepperHolder<T>::~StepperHolder() noexcept(false)
{
    if(steppers != nullptr)
        cudaSafeCall(cudaFree(steppers));
    if(work != nullptr)
        cudaSafeCall(cudaFree(work));
}

/**
 * Runge-Kutta Stepper (t, t+h) with a cuda thread block
 **/
template <class T>
template <class OdeSystem>
__device__
void Stepper<T>::integrate(
    const cg::thread_block& cta,
    const int system_id,
    const T t0, // start time
    const T h, // time span
    OdeSystem& ode  // dyft ode function
    )
{
    // get the cuda thread information
    auto tid = cta.thread_rank(); // patch_id
    auto block_size = cta.size();

    //*** stage 1 - k1 = f(t0, y0) pre-calculated or copied

    //*** stage 2 - k2 = f(t0+c2*h, y0+h*a21*k1)
    // compute y0+h*a21*k1
	// this loop is needed since the #patches could be larger than #threads
	auto yn2 = [=] (const int i) -> void { yn[i] = y0[i]+ h*static_cast<T>(0.2)*k1[i]; };
    cuda::detail::block_process<T, decltype(yn2)>(system_size, yn2);
    // wait till all threads have updated
	cta.sync();
    // compute k2 for each patch
    ode.dydt_block(cta, system_id, t0+static_cast<T>(0.2)*h, yn, k2);
    cta.sync();

    //*** stage 3 - k3 = f(t0+c3*h, y0+h*a31*k1+h*a32*k2)
    // compute y0+h*a31*k1+h*a32*k2
    auto yn3 = [=] (const int i) -> void
    {
        yn[i] = y0[i]
	            + h*(static_cast<T>(3.0/40.0)*k1[i]
	                +static_cast<T>(9.0/40.0)*k2[i]);
	};
    cuda::detail::block_process<T, decltype(yn3)>(system_size, yn3);
	cta.sync();
    // compute k3 for each patch
    ode.dydt_block(cta, system_id, t0+static_cast<T>(0.3)*h, yn, k3);
    cta.sync();

    //*** stage 4 - k4 = f(t0+c4*h, y0+h*a41*k1+h*a42*k2+h*a43*k3)
    // compute y0+h*a41*k1+h*a42*k2+h*a43*k3
    auto yn4 = [=] (const int index) -> void {
        yn[index] = y0[index]
            + h*(static_cast<T>(44.0/45.0)*k1[index]
                +static_cast<T>(-56.0/15.0)*k2[index]
                +static_cast<T>(32.0/9.0)*k3[index]
                );
	};
	cuda::detail::block_process<T, decltype(yn4)>(system_size, yn4);
	cta.sync();
    // compute k4 for each patch
    ode.dydt_block(cta, system_id, t0+static_cast<T>(0.8)*h, yn, k4);
    cta.sync();

    //*** stage 5 - k5 = f(t0+c5*h, y0+h*a51*k1+...)
    // compute y0+h*a51*k1+...
    auto yn5 = [=] (const int index) -> void {
        yn[index] = y0[index]
            + h*(static_cast<T>(19372.0/6561.0)*k1[index]
                +static_cast<T>(-25360.0/2187.0)*k2[index]
                +static_cast<T>(64448.0/6561.0)*k3[index]
                +static_cast<T>(-212.0/729.0)*k4[index]
                );
	};
	cuda::detail::block_process<T, decltype(yn5)>(system_size, yn5);
	cta.sync();
    // compute k5 for each patch
    ode.dydt_block(cta, system_id, t0+static_cast<T>(8.0/9.0)*h, yn, k5);
    cta.sync();

    //*** stage 6 - k6 = f(t0+h, y0+h*a61*k1+...)
    // compute y0+h*a61*k1+...
    auto yn6 = [=] (const int index) -> void {
        yn[index] = y0[index]
            + h*(static_cast<T>(9017.0/3168.0)*k1[index]
                +static_cast<T>(-355.0/33.0)*k2[index]
                +static_cast<T>(46732.0/5247.0)*k3[index]
                +static_cast<T>(49.0/176.0)*k4[index]
                +static_cast<T>(-5103.0/18656.0)*k5[index]
                );
    };
    cuda::detail::block_process<T, decltype(yn6)>(system_size, yn6);
	cta.sync();
    // compute k6 for each patch
    ode.dydt_block(cta, system_id, t0+h, yn, k6);
    cta.sync();

    //*** last stage
    // compute y(t0+h)
    auto yn7 = [=] (const int index) -> void {
        yn[index] = y0[index]
            + h*(static_cast<T>(35.0/384.0)*k1[index]
                +static_cast<T>(500.0/1113.0)*k3[index]
                +static_cast<T>(125.0/192.0)*k4[index]
                +static_cast<T>(-2187.0/6784.0)*k5[index]
                +static_cast<T>(11.0/84.0)*k6[index]
                );
    };
    cuda::detail::block_process<T, decltype(yn7)>(system_size, yn7);
	cta.sync();
    // compute k7 = f(t0+h, y(t0+h))
    ode.dydt_block(cta, system_id, t0+h, yn, k7);
    cta.sync();

    // compute the error estimate yn-yn*
    auto enf = [=] (const int index) -> void {
        en[index] =
              h*(static_cast<T>(71.0/57600.0)*k1[index]
                +static_cast<T>(-71.0/16695.0)*k3[index]
                +static_cast<T>(71.0/1920.0)*k4[index]
                +static_cast<T>(-17253.0/339200.0)*k5[index]
                +static_cast<T>(22.0/525.0)*k6[index]
                +static_cast<T>(-1.0/40.0)*k7[index]
                );
	};
	cuda::detail::block_process<T, decltype(enf)>(system_size, enf);
	cta.sync();
    // all done
}

} // end of namespace cuda::ode::dopri5

#endif //
// end of file
