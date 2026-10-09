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

    // anderson acceleration of the cycle map: the iterates kept (0 turns it off) and the mixing
    static constexpr int MAX_DEPTH = 8;
    int depth;
    T beta;
    // the differences of the residuals and of the mapped states [depth, system_size], the last
    // residual and mapped state [system_size]
    T * dF;
    T * dG;
    T * fprev;
    T * gprev;
    int count; // the differences kept so far
    int head; // the slot the next difference goes to
    bool primed; // whether fprev and gprev hold the last cycle
    T rprev; // the last weighted residual

    // methods
    // initialize device, one system per thread
    __device__ void init (const int system_size_, const T atol_, const T rtol_, T * y_,
        const int depth_ = 0, const T beta_ = 1, T * work_ = nullptr)
    {
        atol = atol_;
        rtol = rtol_;
        yold = y_;
        system_size = system_size_;
        depth = work_ == nullptr ? 0 : depth_;
        beta = beta_;
        dF = work_;
        dG = depth > 0 ? work_ + depth * system_size : nullptr;
        fprev = depth > 0 ? work_ + 2 * depth * system_size : nullptr;
        gprev = depth > 0 ? work_ + (2 * depth + 1) * system_size : nullptr;
        count = head = 0;
        primed = false;
    };

    // forget the cycles of an earlier solve
    __device__ void anderson_reset(const cg::thread_block & cta)
    {
        if(cta.thread_rank() == 0)
        {
            count = head = 0;
            primed = false;
        }
        cta.sync();
    };

    // a sum over {n} terms {func(i)}, returned to every thread of the block
    template <class FuncType>
    __device__ T block_sum(const cg::thread_block & cta, const int n, FuncType func)
    {
        __shared__ T total;
        T mine = 0;
        for(int i = cta.thread_rank(); i < n; i += cta.size())
            mine += func(i);
        auto tile = cg::tiled_partition<32>(cta);
        T part = cg::reduce(tile, mine, cg::plus<T>());
        cta.sync();
        if(cta.thread_rank() == 0)
            total = 0;
        cta.sync();
        if(tile.thread_rank() == 0)
            atomicAdd(&total, part);
        cta.sync();
        T result = total;
        cta.sync();
        return result;
    };

    // the start of the next cycle, extrapolated from the last ones; on entry {y} and {yn} hold the
    // end of this cycle, g, and yold its start, x, over the {n} checked components; on exit {y}
    // and {yn} hold the next start: x + beta f - sum_a gamma_a (dG_a - dF_a + beta dF_a), f = g - x,
    // with gamma minimizing the weighted |f - sum_a gamma_a dF_a|, weighted as the convergence test
    __device__ void accelerate(const cg::thread_block & cta, T* y, T* yn, const int n)
    {
        __shared__ T A[MAX_DEPTH][MAX_DEPTH];
        __shared__ T b[MAX_DEPTH];
        __shared__ T gamma[MAX_DEPTH];
        __shared__ bool plain;
        auto x = yold;
        auto weight = [=] (const int i) { return static_cast<T>(1) / (atol + rtol * max(abs(y[i]), abs(x[i]))); };

        // the weighted residual; restart if it grew a lot, the history no longer describing the map
        auto r = cuda::detail::max_block<T>(cta, n, [=] (const int i) { return abs(y[i] - x[i]) * weight(i); });
        if(cta.thread_rank() == 0 && primed && !(r <= 10 * rprev))
        {
            count = head = 0;
            primed = false;
        }
        cta.sync();

        // the differences to the last cycle, then this cycle as the last one
        if(primed)
        {
            for(int i = cta.thread_rank(); i < n; i += cta.size())
            {
                dF[head * system_size + i] = (y[i] - x[i]) - fprev[i];
                dG[head * system_size + i] = y[i] - gprev[i];
            }
        }
        cta.sync();
        for(int i = cta.thread_rank(); i < n; i += cta.size())
        {
            fprev[i] = y[i] - x[i];
            gprev[i] = y[i];
        }
        if(cta.thread_rank() == 0)
        {
            if(primed)
            {
                head = (head + 1) % depth;
                count = min(count + 1, depth);
            }
            primed = true;
            rprev = r;
        }
        cta.sync();
        // nothing to extrapolate from yet: the plain step, y already holds g
        auto m = count;
        if(m == 0)
            return;

        // the weighted normal equations
        for(int a = 0; a < m; a++)
        {
            for(int c = 0; c <= a; c++)
            {
                auto dot = block_sum(cta, n, [=] (const int i) {
                    auto w = weight(i);
                    return w * w * dF[a * system_size + i] * dF[c * system_size + i]; });
                if(cta.thread_rank() == 0)
                    A[a][c] = A[c][a] = dot;
            }
            auto dot = block_sum(cta, n, [=] (const int i) {
                auto w = weight(i);
                return w * w * dF[a * system_size + i] * (y[i] - x[i]); });
            if(cta.thread_rank() == 0)
                b[a] = dot;
        }
        cta.sync();

        // solve them by gaussian elimination with partial pivoting, slightly regularized
        if(cta.thread_rank() == 0)
        {
            T scale = 0;
            for(int a = 0; a < m; a++)
                scale = max(scale, A[a][a]);
            for(int a = 0; a < m; a++)
                A[a][a] += static_cast<T>(1e-12) * scale;
            plain = !(scale > 0);
            for(int a = 0; a < m && !plain; a++)
            {
                int pivot = a;
                for(int c = a + 1; c < m; c++)
                    if(abs(A[c][a]) > abs(A[pivot][a]))
                        pivot = c;
                if(!(abs(A[pivot][a]) > 0)) { plain = true; break; }
                for(int c = 0; c < m; c++) { T t = A[a][c]; A[a][c] = A[pivot][c]; A[pivot][c] = t; }
                { T t = b[a]; b[a] = b[pivot]; b[pivot] = t; }
                for(int c = a + 1; c < m; c++)
                {
                    T factor = A[c][a] / A[a][a];
                    for(int d = a; d < m; d++)
                        A[c][d] -= factor * A[a][d];
                    b[c] -= factor * b[a];
                }
            }
            for(int a = m - 1; a >= 0 && !plain; a--)
            {
                T sum = b[a];
                for(int c = a + 1; c < m; c++)
                    sum -= A[a][c] * gamma[c];
                gamma[a] = sum / A[a][a];
                if(!isfinite(gamma[a]))
                    plain = true;
            }
            // a failed solve forgets the history and takes the plain step
            if(plain)
                count = head = 0;
        }
        cta.sync();
        if(plain)
            return;

        // the extrapolated start of the next cycle
        for(int i = cta.thread_rank(); i < n; i += cta.size())
        {
            auto f = y[i] - x[i];
            auto next = x[i] + beta * f;
            for(int a = 0; a < m; a++)
                next -= gamma[a] * (dG[a * system_size + i] + (beta - 1) * dF[a * system_size + i]);
            y[i] = next;
            yn[i] = next;
        }
        cta.sync();
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
            // printf("inside spinup controller err: %g  convergence: %d yn[0]: %g\n", err, converge, ynew_check[0]);
        }
        cta.sync();
        return converge;
    };

};

template <class T>
__global__ void spinup_controller_init_kernel(
    const int systems_batch, const int system_size,
    const T atol, const T rtol,
    T* yolds, SpinupController<T>* controllers,
    const int depth = 0, const T beta = 1, T* work = nullptr)
{
    // get the thread id as system id
    auto system = threadIdx.x + blockIdx.x*blockDim.x;
    if(system < systems_batch)
    {
        // initialize controller for each system
        auto yold = yolds + system*system_size;
        auto & controller = controllers[system];
        auto mine = work == nullptr ? nullptr : work + system * (2 * depth + 2) * system_size;
        controller.init(system_size, atol, rtol, yold, depth, beta, mine);
    }
}

template <class T>
struct SpinupControllerHolder {
    T atol;
    T rtol;
    int systems_batch;
    int system_size;
    T* yolds;
    T* work; // the anderson work space, [systems_batch, 2 * depth + 2, system_size]
    SpinupController<T> * controllers;

    SpinupControllerHolder(const int systems_batch_, const int system_size_,
        const T atol_, const T rtol_, const int depth = 0, const T beta = 1)
        : systems_batch(systems_batch_), system_size(system_size_), atol(atol_), rtol(rtol_)
    {
        // allocate data to save copies of old y
        cudaSafeCall(cudaMalloc(&yolds, systems_batch*system_size*sizeof(T)));
        // and the anderson work space, if accelerating
        auto kept = min(max(depth, 0), SpinupController<T>::MAX_DEPTH);
        work = nullptr;
        if(kept > 0)
            cudaSafeCall(cudaMalloc(&work, static_cast<size_t>(systems_batch)*(2*kept+2)*system_size*sizeof(T)));
        // allocate device controllers for each system
        cudaSafeCall(cudaMallocManaged(&controllers, systems_batch*sizeof(SpinupController<T>)));
        // initialize each device controller
        int threads = 256;
        int blocks = (systems_batch-1+threads)/threads; // idivup
        spinup_controller_init_kernel<T><<<blocks, threads>>>(systems_batch, system_size, atol, rtol, yolds, controllers,
            kept, beta, work);
        cudaCheckError("spinup_controller_init_kernel error");
    };
    ~SpinupControllerHolder() noexcept(false)
    {
        if(controllers != nullptr)
            cudaSafeCall(cudaFree(controllers));
        if(work != nullptr)
            cudaSafeCall(cudaFree(work));
        if(yolds != nullptr)
            cudaSafeCall(cudaFree(yolds));
    };
};

} // end of namespace cuda::ode::dopri5

#endif // cuda_ode_dopri5_spinup_controller_cuh
// end of file
