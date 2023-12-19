// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2022 - 2023 california institute of technology
// all rights reserved

/**
 * controller.cuh
 * error controller - adjust the step size to achieve required accuracy
 **/

// code guard
#ifndef cuda_ode_dopri5_controller_cuh
#define cuda_ode_dopri5_controller_cuh

#include "external.h"
#include "stepper.cuh"

namespace cuda::ode::dopri5 {

// struct to keep Error Control process data for solving ODE within a thread block
template<class T>
struct __ALIGNED__ Controller
{
    // member variables
    bool accept;
    bool reject;
    T atol;
    T rtol;
    T hnext; // h for next step
    T errold;

    T t0; // t0 is shifting ->t0 + h after each step
    T t1; // t1 the end
    T hrun; // h for the current/previous step
    bool converged;
    bool t1reached;

    int counter;

    // methods
    __device__ void init (T atol_, T rtol_);
    __device__ void check_convergence(const cg::thread_block & cta, Stepper<T>& s);
    template <class OdeFuncType, class... Args>
    __device__ T find_initial_step(const cg::thread_block & cta, const int system_id,
        Stepper<T>& s, OdeFuncType& func, Args... args);

    __device__ void init_run(const T t0_, const T t1_)
    {
        t0 = t0_;
        t1 = t1_;
        converged = false;
        t1reached = false;
        reject = false;
        errold = static_cast<T>(1e-4);
        counter = 0;
    }

    // use a simple version for now
    // TBD: implement find_initial_step
    __device__ void set_init_h()
    {
        // a small number for exponential functions
        hnext = static_cast<T>(0.001)*(t1-t0);
    };
    __device__ void check_reach_t1()
    {
        if(t0+hnext >= t1) {
            t1reached = true;
            hnext = t1-t0;
        }
    }
    __device__ void t0_increment()
    {
        t0 += hrun;
    }

    __device__ void debug_info()
    {
        printf("step controller debug %d %d %d %g %g %g %g %g\n",
                   counter, converged, t1reached, t0, t1, hnext, hrun, errold);
    }

};



template <class T>
struct ControllerHolder {
    T atol; // error tolerance
    T rtol;
    int systems_batch; // number of systems_batch
    Controller<T> * controllers; // device controllers [systems_batch]

    __host__ void set_tolerance(const T atol, const T rtol);

    ControllerHolder(const int systems_batch_, const T atol_=1e-6, const T rtol_=1e-3)
        : systems_batch(systems_batch_)
    {
        cudaSafeCall(cudaMallocManaged(&controllers, systems_batch*sizeof(Controller<T>)));
        set_tolerance(atol_, rtol_);
    }
    ~ControllerHolder() noexcept(false)
    {
        if (controllers != nullptr)
            cudaSafeCall(cudaFree(controllers));
    }
};


template <class T>
__device__ void
Controller<T>::init (const T atol_, const T rtol_)
{
    atol = atol_;
    rtol = rtol_;
}

// Check whether the error is within tolerance
// follow numerical recipe implementation
// for PI step control, choose beta = 0.4/k (k=5)
// @param s Stepper for one system, shared by all threads in one block
// @param h stepping distance, defined for each thread
template <class T>
__device__ void Controller<T>::check_convergence(
    const cg::thread_block & cta,
    Stepper<T>& s)
{
	static const T beta = static_cast<T>(0.0); // 0.4/k
	static const T alpha = static_cast<T>(0.2)
	    -beta*static_cast<T>(0.75); // 1/k - 0.75\beta
	static const T minscale = static_cast<T>(0.2);
	static const T maxscale = static_cast<T>(10.0);
	static const T safety = static_cast<T>(0.9);


    // compute error
    auto y0 = s.y0;
    auto yn = s.yn;
    auto yerr = s.en;

    // error function for each element
    auto lambda = [=] (const int i)
    {
        auto val = yerr[i]/(atol + rtol*max(abs(yn[i]), abs(y0[i])));
        return val*val;
    };
    // sum over all elements
    auto val = cuda::detail::sum_block<T, decltype(lambda)>(cta, s.system_size, lambda);
    cta.sync();

    // continue checking with thread 0
	if(cta.thread_rank() == 0)
	{
        // compute the error
	    auto err = sqrt(val/s.system_size);
	    // scale h for next run
        T scale;
        // keep info of current running h step
        hrun = hnext;

        // check estimated error for convergence
        if (err <= static_cast<T>(1.0))
        {
            // the error estimate is within required accuracy
            converged = true;
            reject = false;
            // prepare h for next step
            if (err == 0.0)
                scale = maxscale;
            else
            {
                scale=safety*pow(err,-alpha)*pow(errold,beta);
                if (scale<minscale) scale=minscale;
                if (scale>maxscale) scale=maxscale;
            }
            // check whether h was rejected in the previous run
            if (reject)
                scale = min(scale,static_cast<T>(1.0));

            hnext *= scale;
            errold=max(err,1.0e-4);
        }
        else
        {
            // not converged
            // reduce h for next trial
            scale=max(safety*pow(err,-alpha),minscale);
            hnext *= scale;
            reject = true;
            converged = false;
        }
        //printf("test controller err h scale hnext %g %g %g %g \n", err, h, scale, hnext);
        counter++;
    }
    // sync and broadcast
    cta.sync();
}

/**
 * determine the initial step size - not ready yet
 * follow scipy implementation
 *   [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
 *          Equations I: Nonstiff Problems", Sec. II.4.
 **/

template <class T> template <class OdeFuncType, class... Args>
__device__ T Controller<T>::find_initial_step(
    const cg::thread_block & cta, const int system_id,
    Stepper<T>& s, OdeFuncType& dydt_patch, Args... args)
 {
    __shared__ T d0, d1, h0, h1, d2, h_try;

    auto y0 = s.y0;
    auto y1 = s.yn;
    auto f0 = s.k1;
    auto f1 = s.k7;

    auto N = s.system_size;

    // scale = atol + abs(y0)*rtol
    // d0 = norm (y0/scale)
    auto d0_norm_func = [=] (const int i) -> T
    {
        auto val = y0[i];
        auto scale = atol + abs(val)*rtol;
        val /= scale;
        return val*val;
    };
    d0 = cuda::detail::sum_block<T, decltype(d0_norm_func)>(cta, N, d0_norm_func);

    // d1 = norm(f0 / scale)
    auto d1_norm_func = [=] (const int i) -> T
    {
        auto val = f0[i];
        auto scale = atol + abs(val)*rtol;
        val /= scale;
        return val*val;
    };
    d1 = cuda::detail::sum_block<T, decltype(d1_norm_func)>(cta, N, d1_norm_func);

    if(cta.thread_rank() ==0) {
       if (d0 < 1e-5 || d1 < 1e-5)
            h0 = 1e-6;
       else
            h0 = 0.01 * d0 / d1;
    }
    // all threads wait for h0 result
    cta.sync();


    // y1 = y0 + h0 * direction * f0
    auto y1func = [=] (const int i) -> void
    {
            y1[i] = y0[i] + h0 *f0[i];
    };
    cuda::detail::block_process<T, decltype(y1func)>(cta, N, y1func);
    cta.sync();

    // f1 = fun(t0 + h0 * direction, y1)
    for (auto patch_id = cta.thread_rank(); patch_id<s.patches_per_system;
        patch_id+=cta.size())
        dydt_patch(f1, s.t0+h0, y1, system_id, patch_id, args...);
    cta.sync();

    // d2 = norm((f1 - f0) / scale) / h0
    auto d2_norm_func = [=] (const int i) -> T
    {
        auto val = f1[i]-f0[i];
        auto scale = atol + abs(val)*rtol;
        val /= scale;
        return val*val;
    };
    d2 = cuda::detail::sum_block<T, decltype(d1_norm_func)>(cta, N, d1_norm_func);

    cta.sync();
    if(cta.thread_rank()==0)
    {
        d2 /= h0;
        if (d1 <= 1e-15 && d2 <= 1e-15)
            h1 = max(1e-6, h0 * 1e-3);
        else
            h1 = pow(0.01 / max(d1, d2),  static_cast<T>(1.0 / (4 + 1)));

        h_try = min(100 * h0, h1);
    }
    return h_try;
}

template <class T>
__global__ void controller_init_kernel(const T atol, const T rtol, const int systems_batch,
    Controller<T>* controllers)
{
    // get the thread id as system id
    auto system = threadIdx.x + blockIdx.x*blockDim.x;
    if(system < systems_batch)
    {
        // initialize stepper for each system
        auto & controller = controllers[system];
        controller.init(atol, rtol);
    }
}

template <class T>
__host__ void ControllerHolder<T>::set_tolerance(const T atol_, const T rtol_)
{
    atol = atol_;
    rtol = rtol_;
    // initialize each stepper
    int threads = NTHREADS;
    int blocks = (systems_batch-1+threads)/threads; // idivup
    controller_init_kernel<T><<<blocks, threads>>>(atol, rtol, systems_batch, controllers);
    cudaCheckError("controller_init_kernel error");
}


} // end of namespace cuda::ode::dopri5

#endif // cuda_ode_dopri5_controller_cuh
// end of file
