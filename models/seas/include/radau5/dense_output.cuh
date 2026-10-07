// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2026 california institute of technology
// all rights reserved

/**
 * dense_output.cuh
 * the dense output of the Radau IIA stepper: y(t0 + x h) = y0 + Q0 x + Q1 x^2 + Q2 x^3
 **/

// code guard
#ifndef cuda_ode_radau5_dense_output_cuh
#define cuda_ode_radau5_dense_output_cuh

#include <dopri5/external.h>
#include <dopri5/detail.cuh>
#include "stepper.cuh"

namespace cuda::ode::radau5 {

// the dense output state of one system
template <class T>
struct __ALIGNED__ DenseOutput {
    int neval; // number of time points
    int system_size; // number of y elements
    const T* teval; // [neval]
    T* yeval; // [systems, neval, system_size]
    int current_t_index; // the next t_eval to output

    __device__ void init(const int system_size_, const int neval_, const T* teval_, T* yeval_)
    {
        system_size = system_size_;
        neval = neval_;
        teval = teval_;
        yeval = yeval_;
        current_t_index = 0;
    }

    __device__ void interpolate(const cg::thread_block & cta, const Stepper<T> & s, T* yt,
        const T t0, const T h, const T t)
    {
        auto N = system_size;
        auto x = (t - t0)/h;
        for (auto i = static_cast<int>(cta.thread_rank()); i < N; i += cta.size())
            yt[i] = s.y0[i] + x*(s.Q[i] + x*(s.Q[N + i] + x*s.Q[2*N + i]));
    }

    // compute the dense outputs of the t_eval within [t0, t0+h)
    __device__ void output(const cg::thread_block & cta, Stepper<T> & stepper, const int system_id,
        const T t0, const T h)
    {
        auto t1 = t0 + h;
        if (teval[current_t_index] > t1 || teval[neval-1] < t0)
            return;
        for (int it = current_t_index; it < neval; it++) {
            auto t = teval[it];
            if (t >= t0 && t < t1) {
                auto yout = yeval + ((size_t)system_id*neval + it)*system_size;
                interpolate(cta, stepper, yout, t0, h, t);
                cta.sync();
            }
            else {
                current_t_index = it;
                break;
            }
        }
        // the above iteration doesn't compute teval[neval-1] if it is t1
        if (teval[neval-1] == t1) {
            auto yout = yeval + ((size_t)system_id*neval + neval - 1)*system_size;
            cuda::detail::vector_copy<T>(cta, yout, stepper.yn, system_size);
        }
    }

    __device__ void reset(const cg::thread_block & cta)
    {
        if (cta.thread_rank() == 0)
            current_t_index = 0;
        cta.sync();
    }
};

template <class T>
__global__ void denseoutput_init_kernel(const int systems_batch, const int system_size,
    const int neval, const T* teval, T* yeval, DenseOutput<T>* outputters)
{
    auto system = threadIdx.x + blockIdx.x*blockDim.x;
    if (system < systems_batch)
        outputters[system].init(system_size, neval, teval, yeval);
}

template <class T>
struct DenseOutputHolder {
    int neval;
    int systems_batch;
    int system_size;
    const T* teval; // [neval]
    T* yeval; // [systems, neval, system_size]
    DenseOutput<T>* outputters;

    DenseOutputHolder(const int systems_batch_, const int system_size_,
        const int neval_, const T* teval_, T* yeval_)
        : neval(neval_), systems_batch(systems_batch_), system_size(system_size_),
          teval(teval_), yeval(yeval_)
    {
        cudaSafeCall(cudaMalloc((void **)&outputters, systems_batch*sizeof(DenseOutput<T>)));
        int threads = NTHREADS;
        int blocks = (systems_batch-1+threads)/threads; // idivup
        denseoutput_init_kernel<T><<<blocks, threads>>>(systems_batch, system_size,
            neval, teval, yeval, outputters);
        cudaCheckError("radau5 denseoutput_init_kernel error");
    }

    ~DenseOutputHolder() noexcept(false)
    {
        if (outputters != nullptr)
            cudaSafeCall(cudaFree(outputters));
    }
};

} // end of namespace cuda::ode::radau5

#endif // cuda_ode_radau5_dense_output_cuh
// end of file
