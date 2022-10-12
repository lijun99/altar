// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2022 california institute of technology
// all rights reserved

/**
 * dopri5.cuh
 * Runge-Kutta Dormand Prince 54 ODE solver (currently fixed step)
 **/

// code guard
#ifndef __DOPRI5_CUH__
#define __DOPRI5_CUH__

#include "details.cuh"

// enclosed in namespace
namespace ode::dopri5 {

/**
 * Explicit Runge-Kutta method of order 5(4)
 *    order = 5
 *    error_estimator_order = 4
 *    n_stages = 6
 *    Butcher Tableau
 **/
template<typename T>
struct tableau {
    static const int stage = 6;
    static const int order = 5;
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

/**
 *RK state data
**/

template <typename T>
struct step_state {
    int system_size; // components of y
    T t0; // start t0
    T t1; // end t1
    T h;  // time step
    T* y0; // [size] initial value
    T* yn; // [size] final value at t=t0+h
    T* k1; // f(t0, y0)
    T *k2, *k3, *k4, *k5, *k6;
    T* k7; // f(t0+h, yn)
    __device__ __host__ void init(int, T*);
};

// constructor - to initialize state data
template <typename T>
__device__ __host__
void step_state<T>::init (int n, T* rk_work)
{
    system_size = n;
    // allocate memory for step data
    y0 = rk_work;
    yn = rk_work +    system_size;
    k1 = rk_work +  2*system_size;
    k2 = rk_work +  3*system_size;
    k3 = rk_work +  4*system_size;
    k4 = rk_work +  5*system_size;
    k5 = rk_work +  6*system_size;
    k6 = rk_work +  7*system_size;
    k7 = rk_work +  8*system_size;
}

/**
 * Runge-Kutta Step within (t, t+h)
 **/
template <typename T, typename Func, typename... Args>
__host__ __device__
void rk_step_impl(
        tableau<T>& t, // the Butcher Tableau for rk54
        step_state<T>& s,    // process data for ode integration
        Func dydt,           // dyft ode function
        Args... args         // additional (other than y and t) parameters to call ode function
        )
{
    // get the state data
    int n = s.system_size;
    auto t0 = s.t0; // start time
    auto h = s.h;  // time step
    auto tn = s.t1; // final time
    auto y0 = s.y0; // initial value
    auto yn = s.yn; //final value

    int i;
    // stage 1 - k1 = f(t0, y0) pre-calculated or copied
    // stage 2 - k2 = f(t0+c2*h, y0+h*a21*k1)
	for (i=0; i<n; i++)
    	yn[i] = y0[i] + h*t.a21*s.k1[i];
	dydt(s.k2, t0+t.c2*h, yn, n, args...);
	// stage 3 - k3
	for (i=0; i<n; i++)
		yn[i] = y0[i] + h*(t.a31*s.k1[i]+t.a32*s.k2[i]);
	dydt(s.k3, t0+t.c3*h, yn, n, args...);
	// stage 4
	for (i=0; i<n; i++)
		yn[i] = y0[i] + h*(t.a41*s.k1[i]+t.a42*s.k2[i]+t.a43*s.k3[i]);
	dydt(s.k4, t0+t.c4*h, yn, n, args...);
	// stage 5
	for (i=0; i<n; i++)
		yn[i] = y0[i] + h*(t.a51*s.k1[i]+t.a52*s.k2[i]+t.a53*s.k3[i]+t.a54*s.k4[i]);
	dydt(s.k5, t0+t.c5*h, yn, n, args...);
    // stage 6
	for (i=0; i<n; i++)
		yn[i] = y0[i] + h*(t.a61*s.k1[i]+t.a62*s.k2[i]+t.a63*s.k3[i]+t.a64*s.k4[i]+t.a65*s.k5[i]);
	dydt(s.k6, tn, yn, n, args...);
    // compute the final y(t+h) value
	for (i=0; i<n; i++)
		yn [i] = y0[i] + h*(t.b1*s.k1[i]+t.b3*s.k3[i]+t.b4*s.k4[i]+t.b5*s.k5[i]+t.b6*s.k6[i]);
	// compute the final f(t+h, y(t+h)) value
	dydt(s.k7, tn, yn, n, args...);
    // all done
}

/**
 * dense output data
 **/
template<typename T>
struct interpolator {
    // D coefficient
    static const T d1=-12715105075.0/11282082432.0,
	    d3=87487479700.0/32700410799.0, d4=-10690763975.0/1880347072.0,
	    d5=701980252875.0/199316789632.0, d6=-1453857185.0/822651844.0,
	    d7=69997945.0/29380423.0;
    // r1, .. r4
    T * rcont1; // [system_size]vector
    T * rcont2;
    T * rcont3;
    T * rcont4;
    T * rcont5;
    // save the time information
    T t0;
    T h;
    bool computed; // whether rcont are computed
    // initialize
    __device__ __host__ void init(int, T*);
    // prepare for dense output from the rk step data
    __device__ __host__ void prepare_dense(step_state<T>&);
    // interpolate for a given time t
    __device__ __host__ void interpolate_dense(T* yt, const T t, const int n);
};

template <typename T>
void interpolator<T>::init (int system_size, T* rk_work)
{
    // allocate the work data
    rcont1 = rk_work +  9*system_size;
    rcont2 = rk_work + 10*system_size;
    rcont3 = rk_work + 11*system_size;
    rcont4 = rk_work + 12*system_size;
    rcont5 = rk_work + 13*system_size;
    computed = false;
}


template <typename T>
__device__ __host__
void interpolator<T>::prepare_dense(step_state<T>& s)
{
    // get time from state
    t0 = s.t0;
    h = s.h;
    auto n = s.system_size;

    for (int i=0;i<n;i++) {
		rcont1[i]=s.y0[i];
		auto ydiff=s.yn[i]-s.y0[i];
		rcont2[i]=ydiff;
		auto bspl=h*s.k1[i]-ydiff;
		rcont3[i]=bspl;
		rcont4[i]=ydiff-h*s.k7[i]-bspl;
		rcont5[i]=h*(d1*s.k1[i]+d3*s.k3[i]+d4*s.k4[i]+d5*s.k5[i]+d6*s.k6[i]+d7*s.k7[i]);
	}
	computed = true;
}

template <typename T>
__device__ __host__
void interpolator<T>::interpolate_dense(T* yt, const T t, const int n)
{
    // get the theta (distance)
    auto s=(t-t0)/h;
	auto s1=1.0-s;
	// iterate over system indices
	for(int i=0; i<n; i++)
	    yt[i] = rcont1[i]+s*(rcont2[i]+s1*(rcont3[i]+s*(rcont4[i]+s1*rcont5[i])));
	// all done
}

// this is rather a holder of the output state data
template <typename T>
struct dense_output_state{
    int nout; //number of time points
    int index; //
    const T* tout; // [nout] vector
    T* yout; // [nout, system_size]
    __device__ __host__ void init(const int n, const T* t, T* y)
    {
        nout = n;
        tout = t;
        yout = y;
        index = 0;
    }
};

template <typename T>
__device__ __host__
void dense_output(
    dense_output_state<T>& out_state,
    interpolator<T>& interp,
    step_state<T>& rk_state)
{

  //    int sample = blockIdx.x *blockDim.x + threadIdx.x;
    // get the parameters
    auto system_size = rk_state.system_size;
    auto t0 = rk_state.t0;
    auto t1 = rk_state.t1;
    // auto h = rk_state.h;

    bool is_in_range = 1;
    interp.computed = false;
    auto & index = out_state.index;
    auto nout = out_state.nout;
    while (is_in_range && index < nout){
        auto t = out_state.tout[index];
        // check whether t is \in [t0, t0+h]
        // printf("interpolation check %d %f %f %f %d\n", index, t, t0, t1, t>=t0 && t<=t1);
        if(t>=t0 && t<=t1) {
            // in range, perform the interpolation
            // check whether rcond_n vectors are initialized, if not, prepare them
            // only need to do once for all t's within [t0, t0+h]
            if(!interp.computed){
                interp.prepare_dense(rk_state);
            }
            // get the output y location
            auto yout = out_state.yout + index*system_size;
            // perform the interpolation
            interp.interpolate_dense(yout, t, system_size);
            // printf("interpolation result %d %f %f %f %f \n", index, t, t0, t0+h, yout[0]);
            // move index
            index++;
        }
        else {
            // set in range to false
            is_in_range = 0;
        }
    }
}

template<typename T>
__device__ __host__
void vector_copy(T*a, const T*b, const int size)
{
    for(int i=0; i<size; i++)
        a[i] = b[i];
}

/**
 * Solve one sample of ivp
 * dydt f=dy/dt function
 * t0, t1 start and end time
 * steps number of rk integral steps
 * y0 initial values at t0
 * tout dense output t values
 * yout dense output y values
 **/
template <typename T, typename Func, typename... Args>
__device__ void rk_solver_fixedstep(
    const int rk_steps,
    T* rk_work,
    const int system_size,
    const T t0, const T t1,
    const T* y0,
    bool is_dense_output,
    const T* t_out, T* y_out, const int n_out,
    Func dydt, Args... args
    )
{
    // determine the step size
    auto h_step = (t1-t0)/rk_steps;
    // initialize the tableau and state
    tableau<T> table;
    step_state<T> rk_step_state;
    rk_step_state.init(system_size, rk_work);
    // data for dense output
    dense_output_state<T> out_state;
    interpolator<T> interp;
    // only allocate the
    if(is_dense_output)
    {
        // only initialize them when dense output is desired
        out_state.init(n_out, t_out, y_out);
        interp.init(system_size, rk_work);
    }

    // assign the initial values to state
    vector_copy<T>(rk_step_state.y0, y0, system_size);
    // assign the initial k1 value: f(t0, y0)
    dydt(rk_step_state.k1, t0, y0, system_size, args...);

    // iterate over steps
    for(int step = 0; step<rk_steps; step++)
    {
        // set the t0 and step
        rk_step_state.t0 = t0 + step*(t1-t0)/rk_steps;
        rk_step_state.t1 = t0 + (step+1)*(t1-t0)/rk_steps;
        rk_step_state.h = h_step;
        // printf("step loop %d %f %f\n", step, rk_step_state.t0, rk_step_state.yn[0]);
        // call ode solver
        rk_step_impl<T, Func, Args...>(table, rk_step_state, dydt, args...);
        // if denseoutput
        if(is_dense_output) {
            dense_output<T>(out_state, interp, rk_step_state);
            //if(sample!=0) {
            //    printf("inside rk kernel dense %d %g %g\n", sample, out_state.yout[0], y_out[0]);
            // }
        }
        // printf("step data %f %f\n", rk_step_state.y0[0], rk_step_state.yn[0]);
        // copy final results to next step final values
        vector_copy<T>(rk_step_state.y0, rk_step_state.yn, system_size); // y
        vector_copy<T>(rk_step_state.k1, rk_step_state.k7, system_size); // f(t, y)
    }
    // if not dense_output, we copy the last step y to output
    if (!is_dense_output) {
        vector_copy<T>(y_out, rk_step_state.yn, system_size);
    }
}

// a generic interface for calling the rk solver for a batch of samples
template <typename T, typename Func, typename... Args>
__global__ void rk_solver_fixedstep_batch(
    const int rk_steps,
    const int samples,
    const int system_size,
    const T t0, const T t1,
    const T* y0,
    bool is_dense_output,
    const T* t_out, T* y_out, const int n_out,
    Func dydt, Args... args
    )
{
    // one thread per sample, to get the sample index
    int sample = blockIdx.x *blockDim.x + threadIdx.x;

    // check whether thread id is in range of samples
    // - a common practice for cuda since #threads might be bigger than #samples
    if(sample >= samples)
        return;

    // get the starting pointer for samples
    auto y0_s = y0 + sample*system_size;
    auto yout_s = y_out + sample*system_size*n_out;

    // assume t_out is the same
    // call the ode solver for this sample
    rk_solver_fixedstep<T, Func, Args...>(rk_steps, system_size,
            t0, t1, y0_s, is_dense_output,
            t_out, yout_s, n_out,
            dydt, args...);
    // all done
}

// a generic interface for calling the rk solver for a batch of samples
// with sample dependent parameters (params) and sample-independent parameters (args)
template <typename T, typename Func, typename... Args>
__global__ void rk_solver_batch_params(
    const int rk_steps,
    const int samples,
    const int system_size,
    const T t0, const T t1,
    const T* y0,
    bool is_dense_output,
    const T* t_out, T* y_out, const int n_out,
    Func dydt,
    const T* params, const int parameters,
    Args... args
    )
{
    // one thread per sample, to get the sample index
    int sample = blockIdx.x *blockDim.x + threadIdx.x;

    // check whether thread id is in range of samples
    // - a common practice for cuda since #threads might be bigger than #samples
    if(sample >= samples)
        return;

    // get the starting pointer for samples
    auto y0_s = y0 + sample*system_size;
    auto yout_s = y_out + sample*system_size*n_out;

    // get the sample dependent parameters
    auto params_s = params + sample*parameters;

    // assume t_out is the same
    // call the ode solver for this sample
    rk_solver_fixedstep<T, Func, const T*, const int, Args...>(
        rk_steps,
        system_size,
        t0, t1, y0_s,
        is_dense_output,
        t_out, yout_s, n_out,
        dydt,
        params_s, parameters,
        args...);
    // all done
}

// a generic function to call rk_solver_batch_params
// params are varied parameters for different samples
// args are fixed parameters
template <typename T, typename Func, typename... Args>
void rk_ode_solver(
    const int rk_steps,
    const int samples,
    const int system_size,
    const T t0, const T t1,
    const T* y0,
    const bool is_dense_output,
    const T* t_out, T* y_out, const int n_out,
    Func dydt,
    const T* params, const int parameters,
    Args... args
    )
{

    // compute the number of gpu blocks needed
    const int threadsPerBlock = 128;
    const int numberOfBlocks = (samples-1+threadsPerBlock)/threadsPerBlock; //IDIVUP
    std::cout << numberOfBlocks;

    // @note required for allocating large memory from kernel
    // not working for all gpus - need check
    // also on how to adjust the limit to request
    cudaSafeCall(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024*1024*1024));

    // call ode solver kernel
    rk_solver_batch_params<T, Func, Args...><<<numberOfBlocks, threadsPerBlock>>>(
        rk_steps,
        samples,
        system_size, // system size
        t0, t1,
        y0,
        dense_output, // dense out = true
        t_out, y_out, n_out, // for dense_output
        dydt, // the above are stand parameters to call rk_solver, below are model-depend parameter, included in args...
        parameters, params, // sample dependent parameters
        args ...  // sample independent parameters
        );
    // check errors
    cudaSafeCall(cudaGetLastError());
    // all done
}


} // end of namespace ode::dopri5

#endif // __DOPRI5_CUH__
// end of file
