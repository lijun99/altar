// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2022 - 2023 california institute of technology
// all rights reserved

/**
 * Dense Output for ODE solver
 * struct Interpolator -- device module to compute y(t) by interpolating
 * struct DenseOutputState -- device module to hold the output data
 * struct DenseOutput -- host module to hold Interpolators and DenseOutputStates
 **/

// code guard
#ifndef cuda_ode_dopri5_dense_output_cuh
#define cuda_ode_dopri5_dense_output_cuh

#include "external.h"
#include "stepper.cuh"

namespace cuda::ode::dopri5 {

// DenseOutput state for one system in device
template <class T>
struct __ALIGNED__ DenseOutput{
    // outputs
    int neval; //number of time points
    int system_size; // number of y elements
    const T* teval; // [neval] vector
    T* yeval; // [systems, neval, system_size]

    // temporary data for interpolating
    int current_t_index; // keep record of current t_out
    T * rcont1; // vector [system_size]
    T * rcont2;
    T * rcont3;
    T * rcont4;
    T * rcont5;

    // initialize the pointers
    __device__ void init(const int system_size_,
        const int neval_, const T* teval_, T* yeval_,
        T * rk_work) // run by one thread per sample
    {
        system_size = system_size_;
        neval = neval_;
        teval = teval_;
        yeval = yeval_; // note: this is set to the start of all yeval
        current_t_index = 0;
        rcont1 = rk_work;
        rcont2 = rk_work + system_size;
        rcont3 = rk_work + 2*system_size;
        rcont4 = rk_work + 3*system_size;
        rcont5 = rk_work + 4*system_size;

    };

    __device__ void prepare_dense(const cg::thread_block & cta, Stepper<T>& s, const T h) //run by one block per sample
    {

        for (auto i=cta.thread_rank(); i<system_size; i+=cta.size())
        {
		    rcont1[i]=s.y0[i];
		    auto ydiff=s.yn[i]-s.y0[i];
		    rcont2[i]=ydiff;
		    auto bspl=h*s.k1[i]-ydiff;
		    rcont3[i]=bspl;
		    rcont4[i]=ydiff-h*s.k7[i]-bspl;
		    rcont5[i]=h*(
		        static_cast<T>(-12715105075.0/11282082432.0)*s.k1[i]
		        +static_cast<T>(87487479700.0/32700410799.0)*s.k3[i]
		        +static_cast<T>(-10690763975.0/1880347072.0)*s.k4[i]
		        +static_cast<T>(701980252875.0/199316789632.0)*s.k5[i]
		        +static_cast<T>(-1453857185.0/822651844.0)*s.k6[i]
		        +static_cast<T>(69997945.0/29380423.0)*s.k7[i]);
	    }
	    cta.sync();
    };

    __device__ void interpolate(const cg::thread_block & cta, T* yt,
        const T t0, const T h, const T t)
    {
        // get the theta (distance)
        auto s=(t-t0)/h;
	    auto s1=(T)1.0-s;
	    // iterate over system indices
	    for (int i=cta.thread_rank();i<system_size;i+=cta.size())
	        yt[i] = rcont1[i]+s*(rcont2[i]+s1*(rcont3[i]+s*(rcont4[i]+s1*rcont5[i])));
	    // all done
    };

    // check whether any tevel fall within (t0, t0+h) and if so, compute dense outputs
    __device__ void output(const cg::thread_block & cta, Stepper<T> & stepper, const int system_id,
        const T t0, const T h)
    {

        auto t1 = t0+h;

        // first check whether the interpolation time points overlap
        if(teval[current_t_index] > t1 || teval[neval-1] < t0)
            return; // no overlap
        // otherwise, proceed
        // compute the rconts
        prepare_dense(cta, stepper, h);
        // iterate
        for(int it=current_t_index;  it<neval; it++)
        {
            auto t = teval[it];
            if (t>=t0 && t<=t1) {
                auto yout = yeval + (system_id*neval+it)*system_size;
                interpolate(cta, yout, t0, h, t);
            }
            else {
                current_t_index = it;
                break;
            }
        }
        // all done
    };
    __device__ void reset(const cg::thread_block & cta)
    {
        if(cta.thread_rank() == 0)
        {
            current_t_index = 0;
        }
        cta.sync();
    };
};

// gpu holder of the output state data
// hold the dense output data
template <class T>
struct DenseOutputHolder {
    int neval;
    int systems_batch;
    int system_size;
    // output
    const T* teval; // [neval]
    T* yeval; // [systems, neval, system_size] note system >= systems_batch
    DenseOutput<T>* outputters;
    // interpolators
    T * rconts;

    // constructor
    DenseOutputHolder (const int systems_batch_, const int system_size_,
        const int neval_, const T* teval_, T* yeval_);
    ~DenseOutputHolder();
};

template <class T>
__global__ void denseoutput_init_kernel(const int systems_batch, const int system_size,
    const int neval, const T* teval, T* yeval,
    T* rconts,
    DenseOutput<T>* outputters)
{
    // get the thread id as system id
    auto system = threadIdx.x + blockIdx.x*blockDim.x;
    if(system < systems_batch)
    {
        // initialize stepper for each system
        auto& outputter = outputters[system];
        auto rcont_system = rconts + 5*system_size*system;
        outputter.init(system_size, neval, teval, yeval, rcont_system);
    }
}


// DenseOutput Implementations
template <class T>
DenseOutputHolder<T>::DenseOutputHolder (
    const int systems_batch_, const int system_size_,
    const int neval_, const T* teval_, T* yeval_)
        : neval(neval_), system_size(system_size_), systems_batch(systems_batch_), teval(teval_), yeval(yeval_)
{
    // allocate the interpolator work data
    cudaSafeCall(cudaMalloc(&rconts, 5*system_size*systems_batch*sizeof(T)));

    // allocate the device output states
    cudaSafeCall(cudaMalloc((void **)&outputters, systems_batch*sizeof(DenseOutput<T>)));

    // initialize each stepper
    int threads = NTHREADS;
    int blocks = (systems_batch-1+threads)/threads; // idivup
    denseoutput_init_kernel<T><<<blocks, threads>>>(systems_batch, system_size,
        neval, teval, yeval, rconts, outputters);
    cudaCheckError("denseoutput_init_kernel error");
}

// destructor
template <class T>
DenseOutputHolder<T>::~DenseOutputHolder()
{
    if (outputters!=nullptr)
        cudaSafeCall(cudaFree(outputters));
    if(rconts != nullptr)
        cudaSafeCall(cudaFree(rconts));
}

} // end of namespace cuda::ode::dopri5

#endif // cuda_ode_dopri5_dense_output_cuh
// end of file
