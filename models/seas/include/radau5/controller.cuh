// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2026 california institute of technology
// all rights reserved

/**
 * controller.cuh
 * step size control of the Radau IIA stepper, following scipy.integrate.Radau
 **/

// code guard
#ifndef cuda_ode_radau5_controller_cuh
#define cuda_ode_radau5_controller_cuh

#include <dopri5/external.h>
#include <dopri5/detail.cuh>
#include "stepper.cuh"

namespace cuda::ode::radau5 {

template<class T>
struct __ALIGNED__ Controller
{
    bool reject;
    T atol;
    T rtol;
    T hnext; // h for next step

    T t0; // t0 is shifting ->t0 + h after each step
    T t1; // t1 the end
    T hrun; // h for the current/previous step
    bool converged; // whether the current step is accepted
    bool t1reached;
    int counter;

    // the previous accepted step, for the step size prediction
    T h_start; // the step proposed at the start of the current step
    T h_old; // the one proposed at the start of the previous step
    T err_old; // and its error
    bool fresh; // whether no step has been tried since the last accepted one

    // step statistics, over one solve
    int accepted;
    int rejected;
    int stiff; // always zero: Radau is not stability limited
    int cycles;
    bool failed; // whether the system was given up on
    // the most steps, accepted or rejected, in one solve
    static constexpr int MAX_STEPS = 1000000;
    // the most rejections in a row; each shrinks the step by half or more
    static constexpr int MAX_REJECTIONS = 50;
    int consecutive_rejections;
    T hmin;
    T hmax;

    __device__ void init(const T atol_, const T rtol_)
    {
        atol = atol_;
        rtol = rtol_;
    }

    __device__ void init_run(const T t0_, const T t1_)
    {
        t0 = t0_;
        t1 = t1_;
        converged = false;
        t1reached = false;
        reject = false;
        counter = 0;
        h_old = static_cast<T>(0);
        err_old = static_cast<T>(0);
        fresh = true;
    }

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

    __device__ void reset_statistics()
    {
        accepted = 0;
        rejected = 0;
        stiff = 0;
        cycles = 0;
        failed = false;
        consecutive_rejections = 0;
        hmin = cuda::std::numeric_limits<T>::max();
        hmax = static_cast<T>(0);
    }

    __device__ void debug_info()
    {
        printf("radau5 controller %d %d %d %d %g %g %g %g\n",
               blockIdx.x, counter, converged, t1reached, t0, t1, hnext, hrun);
    }

    // the step size factor from the error, following scipy's {predict_factor}
    __device__ T predict_factor(const T h, const T err) const
    {
        auto multiplier = static_cast<T>(1);
        if (err_old > static_cast<T>(0) && h_old > static_cast<T>(0) && err > static_cast<T>(0))
            multiplier = h/h_old*pow(err_old/err, static_cast<T>(0.25));
        if (err == static_cast<T>(0))
            return cuda::std::numeric_limits<T>::max();
        return min(static_cast<T>(1), multiplier)*pow(err, static_cast<T>(-0.25));
    }

    /**
     * the initial step, after an event or at the start; the stepper starts over too
     * follows scipy, for an error estimate of order 3:
     *   [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
     *          Equations I: Nonstiff Problems", Sec. II.4.
     **/
    template <class OdeSystem>
    __device__ void select_initial_step(const cg::thread_block & cta, const int system_id,
        Stepper<T>& s, OdeSystem& ode)
    {
        __shared__ T d0, d1, d2, h0;
        auto N = s.system_size;

        // the stepper starts over, with my tolerances
        s.reset(cta);
        if (cta.thread_rank() == 0) {
            s.atol = atol;
            s.rtol = rtol;
            s.newton_tol = max(static_cast<T>(10)*cuda::std::numeric_limits<T>::epsilon()/rtol,
                               min(static_cast<T>(0.03), sqrt(rtol)));
        }
        cta.sync();
        s.set_f0_value(cta, t0, system_id, ode);

        auto y0 = s.y0;
        auto f0 = s.f0;
        auto scale = s.scale;
        auto y1 = s.ys;
        auto f1 = s.fs;
        auto a = atol, r = rtol;
        auto scale_func = [=] (const int i) -> void { scale[i] = a + abs(y0[i])*r; };
        cuda::detail::block_process<T, decltype(scale_func)>(cta, N, scale_func);
        cta.sync();

        auto d0_func = [=] (const int i) -> T { auto v = y0[i]/scale[i]; return v*v; };
        auto sd0 = cuda::detail::sum_block<T, decltype(d0_func)>(cta, N, d0_func);
        auto d1_func = [=] (const int i) -> T { auto v = f0[i]/scale[i]; return v*v; };
        auto sd1 = cuda::detail::sum_block<T, decltype(d1_func)>(cta, N, d1_func);
        if (cta.thread_rank() == 0) {
            d0 = sqrt(sd0/N);
            d1 = sqrt(sd1/N);
            h0 = (d0 < static_cast<T>(1e-5) || d1 < static_cast<T>(1e-5)) ? static_cast<T>(1e-6) : static_cast<T>(0.01)*d0/d1;
            h0 = min(h0, t1 - t0);
        }
        cta.sync();

        auto hh = h0;
        auto y1_func = [=] (const int i) -> void { y1[i] = y0[i] + hh*f0[i]; };
        cuda::detail::block_process<T, decltype(y1_func)>(cta, N, y1_func);
        cta.sync();
        ode.dydt_block(cta, system_id, t0 + hh, y1, f1);
        cta.sync();

        auto d2_func = [=] (const int i) -> T { auto v = (f1[i] - f0[i])/scale[i]; return v*v; };
        auto sd2 = cuda::detail::sum_block<T, decltype(d2_func)>(cta, N, d2_func);
        if (cta.thread_rank() == 0) {
            d2 = sqrt(sd2/N)/h0;
            T h1;
            if (d1 <= static_cast<T>(1e-15) && d2 <= static_cast<T>(1e-15))
                h1 = max(static_cast<T>(1e-6), h0*static_cast<T>(1e-3));
            else
                h1 = pow(static_cast<T>(0.01)/max(d1, d2), static_cast<T>(0.25));
            hnext = min(min(static_cast<T>(100)*h0, h1), t1 - t0);
        }
        cta.sync();
    }

    // accept or reject the step the stepper just tried, and propose the next one
    __device__ void check_convergence(const cg::thread_block & cta, Stepper<T>& s)
    {
        using namespace coefficients;
        constexpr auto MIN_FACTOR = static_cast<T>(0.2);
        constexpr auto MAX_FACTOR = static_cast<T>(10);

        if (cta.thread_rank() == 0) {
            hrun = hnext;
            if (fresh) {
                h_start = hrun;
                fresh = false;
            }
            if (!s.newton_converged) {
                // the iterations failed: halve the step
                hnext *= static_cast<T>(0.5);
                s.lu_valid = false;
                reject = true;
            }
            else {
                auto err = s.error_norm;
                auto safety = static_cast<T>(0.9)*(2*NEWTON_MAXITER + 1)/(2*NEWTON_MAXITER + s.n_iter);
                if (err > static_cast<T>(1)) {
                    // too inaccurate: shrink the step
                    auto factor = predict_factor(hrun, err);
                    hnext *= max(MIN_FACTOR, safety*factor);
                    s.lu_valid = false;
                    reject = true;
                }
                else {
                    // accepted; keep the factorizations if the step size barely changes
                    auto recompute = s.n_iter > 2 && s.rate > static_cast<T>(1e-3);
                    auto factor = min(MAX_FACTOR, safety*predict_factor(hrun, err));
                    if (!recompute && factor < static_cast<T>(1.2))
                        factor = static_cast<T>(1);
                    else
                        s.lu_valid = false;
                    s.recompute_jac = recompute;
                    h_old = h_start;
                    err_old = err;
                    fresh = true;
                    hnext = hrun*factor;
                    reject = false;
                    converged = true;
                    accepted++;
                    consecutive_rejections = 0;
                    hmin = min(hmin, hrun);
                    hmax = max(hmax, hrun);
                }
            }
            if (reject) {
                converged = false;
                // a shorter step no longer reaches t1
                t1reached = false;
                rejected++;
                consecutive_rejections++;
            }
            s.rejected = reject;
            // give up on a system whose steps keep being rejected, e.g., after its state
            // overflowed, or that needs more steps than any sensible solve
            if (!isfinite(hnext) || (t0 < t1 && hnext < cuda::std::numeric_limits<T>::min())
                || consecutive_rejections >= MAX_REJECTIONS || accepted + rejected >= MAX_STEPS) {
                failed = true;
                converged = true;
                t1reached = true;
            }
            counter++;
        }
        cta.sync();
    }

    // after an accepted step: its dense output, and what the next step can reuse
    __device__ void record_step(const cg::thread_block & cta, Stepper<T>& s)
    {
        using namespace coefficients;
        auto N = s.system_size;
        auto Z = s.Z;
        auto Q = s.Q;
        // Q = Z^T P
        for (auto i = static_cast<int>(cta.thread_rank()); i < N; i += cta.size()) {
            auto z0 = Z[i], z1 = Z[N + i], z2 = Z[2*N + i];
            Q[i] = static_cast<T>(P00)*z0 + static_cast<T>(P10)*z1 + static_cast<T>(P20)*z2;
            Q[N + i] = static_cast<T>(P01)*z0 + static_cast<T>(P11)*z1 + static_cast<T>(P21)*z2;
            Q[2*N + i] = static_cast<T>(P02)*z0 + static_cast<T>(P12)*z1 + static_cast<T>(P22)*z2;
        }
        if (cta.thread_rank() == 0) {
            s.h_prev = hrun;
            s.pred_valid = true;
            // y0 moves to yn
            s.f0_valid = false;
            s.rejected = false;
            if (s.recompute_jac)
                s.jac_valid = false;
            else
                s.jac_current = false;
        }
        cta.sync();
    }
};


template <class T>
__global__ void controller_init_kernel(const T atol, const T rtol, const int systems_batch,
    Controller<T>* controllers)
{
    auto system = threadIdx.x + blockIdx.x*blockDim.x;
    if(system < systems_batch)
        controllers[system].init(atol, rtol);
}

template <class T>
struct ControllerHolder {
    T atol;
    T rtol;
    int systems_batch;
    Controller<T> * controllers; // managed, so the host can read the statistics

    ControllerHolder(const int systems_batch_, const T atol_=1e-6, const T rtol_=1e-3)
        : atol(atol_), rtol(rtol_), systems_batch(systems_batch_)
    {
        cudaSafeCall(cudaMallocManaged(&controllers, systems_batch*sizeof(Controller<T>)));
        int threads = NTHREADS;
        int blocks = (systems_batch-1+threads)/threads; // idivup
        controller_init_kernel<T><<<blocks, threads>>>(atol, rtol, systems_batch, controllers);
        cudaCheckError("radau5 controller_init_kernel error");
    }
    ~ControllerHolder() noexcept(false)
    {
        if (controllers != nullptr)
            cudaSafeCall(cudaFree(controllers));
    }
};

} // end of namespace cuda::ode::radau5

#endif // cuda_ode_radau5_controller_cuh
// end of file
