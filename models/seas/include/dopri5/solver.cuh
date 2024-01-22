// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2022 - 2023 california institute of technology
// all rights reserved

/**
 * dopri5.cuh
 * Runge-Kutta Dormand Prince 54 ODE solver
 **/

// code guard
#ifndef cuda_ode_dopri5_solver_cuh
#define cuda_ode_dopri5_solver_cuh

#include "external.h"
#include "stepper.cuh"
#include "controller.cuh"
#include "dense_output.cuh"
#include "events.cuh"
#include <assert.h>


// enclosed in namespace
namespace cuda::ode::dopri5 {

// declaration of the dopri5 solver class`
template <class real_type, class ode_system_type, class event_type = FixedEvents<real_type>>
struct Solver
{
    // rename types
    using size_type = std::size_t;
    using stepper_type = cuda::ode::dopri5::Stepper<real_type>;
    using stepper_holder_type = cuda::ode::dopri5::StepperHolder<real_type>;
    using controller_type = cuda::ode::dopri5::Controller<real_type>;
    using controller_holder_type = cuda::ode::dopri5::ControllerHolder<real_type>;
    using output_type = cuda::ode::dopri5::DenseOutput<real_type>;
    using output_holder_type = cuda::ode::dopri5::DenseOutputHolder<real_type>;

    // variables
    int systems_batch; // Batch of systems allocated, each system is processed by one block
    int system_size; // number of elements in one system, patches*units
    int threads; // number of threads to use for each system (one thread block)

    ode_system_type ode; // define the ode system
    event_type events; // define the event

    // processors
    controller_holder_type * controller_holder;
    stepper_holder_type * stepper_holder;
    output_holder_type * output_holder;

    // constructor
    Solver(ode_system_type& o, event_type& e, const real_type atol, const real_type rtol,
           const int systems_batch, const int threads);

    // set initial y(t0) values
    // @note if there are many batches, this needs to be called multiple times
    // @note y0 is for all systems, including when use_y0_for_all = true
    void set_init_values(const real_type* y0, const bool use_y0_for_all, const int systems, const int system_offset);
    // a default call option
    void set_init_values(const real_type* y0, const bool use_y0_for_all) { set_init_values(y0, use_y0_for_all, systems_batch, 0);};
    // set dense output
    // @note called once, since yevals save output for all systems
    void set_dense_output(const int neval, const real_type * tevals, real_type * yevals);
    // solve the ivp
    void solve_ivp(const bool dense_out, const int systems, const int system_offset);
    // a default call option
    void solve_ivp(const bool dense_out) { solve_ivp(dense_out, systems_batch, 0); };
};


// solver constructor
template <class real_type, class ode_system_type, class event_type>
Solver<real_type, ode_system_type, event_type>::Solver(ode_system_type & ode_, event_type & events_,
    const real_type atol=1e-8, const real_type rtol=1e-6, const int systems_batch_=8192, const int threads_=0)
    : ode(ode_), events(events_)
{
    // if the total number of systems is smaller than provide batch, use real number of systems instead
    systems_batch = min(ode.systems, systems_batch_);
    system_size = ode.system_size;
    stepper_holder = new stepper_holder_type(ode.patches, ode.units, systems_batch);
    controller_holder = new controller_holder_type(systems_batch, atol, rtol);

    // set number of patches
    if (threads_ <= 0 || threads_ > 1024) // keep default values based on number of patches
    {
        auto patches = ode.patches;
        if (patches < 64)
            threads = 32;
        else if (patches < 128)
            threads = 64;
        else if (patches < 256)
            threads =128;
        else if (patches < 512)
            threads = 256;
        else if (patches < 1024 )
            threads = 512;
        else
            threads = 1024;
        // printf("Solver initialized with %i threads based on %i patches\n", threads, patches);
    }
    else
    {
        threads = threads_;
        // printf("Solver initialized with user-defined %i threads\n", threads);
    }
}

// cuda kernel for setting initial values y0
template <class real_type, class ode_system_type, class event_type>
__global__ void set_init_values_kernel(
    const int system_offset,
    ode_system_type ode, // note no reference on global calls
    event_type events,
    Stepper<real_type> * steppers,
    const real_type* y0, const bool use_y0_for_all)
{
    auto block_id = blockIdx.x;
    auto system_id = block_id + system_offset;
    auto cta = cg::this_thread_block();

    // get stepper for this system
    auto& stepper = steppers[block_id];
    // get y0 for this system
    const real_type* y0p;
    if(use_y0_for_all)
        y0p = y0;
    else
        y0p = y0+system_id*ode.system_size;
    // get t0
    auto tevents = events.get_events_time(system_id);
    auto t0 = tevents[0];
    // set y0 and compute f(t0, y0)
    stepper.set_init_value(cta, t0, y0p, system_id, ode);
}

// set initial y0 values
// @use_y0_for_all whether y0 is for all systems
// @parameter y0 input [system_size] or [systems, system_size] depending on use_y0_for_all
template <class real_type, class ode_system_type, class event_type>
void Solver<real_type, ode_system_type, event_type>::set_init_values(
    const real_type* y0, const bool use_y0_for_all, const int systems, const int system_offset=0)
{
    // printf("    inside solver.cuh:set_init_values\n");

    int blocks = systems;
    // printf("      blocks=%i\n", blocks);
    // printf("      system_offset=%i\n", system_offset);
    set_init_values_kernel<real_type, ode_system_type, event_type><<<blocks, threads>>>(
        system_offset,
        ode,
        events,
        stepper_holder->steppers,
        y0, use_y0_for_all);
    cudaCheckError("set_init_values_kernel error");
    // all done
}

template <class real_type, class ode_system_type, class event_type>
__device__ void solve_device( const cg::thread_block & cta,
    const int system_id, const bool dense_out,
    ode_system_type & ode,
    event_type &  events,
    Stepper<real_type> & stepper,
    Controller<real_type> &  controller,
    DenseOutput<real_type> & outputter,
    const bool verbose
    )
{

    outputter.reset(cta);

    // iterate over events
    for(auto it=0; it<events.nevents-1; it++)
    {
        // get the event times and initialize controller
        if(cta.thread_rank()==0)
        {
            auto tevents = events.get_events_time(system_id);
            controller.init_run(tevents[it], tevents[it+1]);
            controller.set_init_h();

            if (verbose)
                printf(".");
        }
        cta.sync();



        // set events at t0
        events.set_events_block(cta, system_id, it, stepper.y0);
        cta.sync();

        // determine a step to start
        controller.select_initial_step(cta, system_id, stepper, ode);
        cta.sync();

        // Safe option: to ask stepper compute anyway, no need to call set_f0
        // need to recompute f0 = f(t0, y0) due to the possible y0 update
        // stepper.set_f0_value(cta, controller.t0, system_id, ode);
        // cta.sync();

        // adaptive steps from t0 to t1
        while(!controller.t1reached)
        {
            if(cta.thread_rank()==0)
            {
                // check whether t0+h >= t1
                // if so, set t1reached=true
                controller.check_reach_t1();
                // reset converged flag
                controller.converged = false;
            }
            cta.sync();

            // adjust step length to reach convergence

            while(!controller.converged)
            {
                // integrate over step h
                auto t0 = controller.t0;
                auto h = controller.hnext;
                stepper.integrate(cta, system_id, t0, h, ode);
                // check the convergence and propose a new step hnext
                controller.check_convergence(cta, stepper);

                // to check the adaptive steps
                // if(cta.thread_rank()==0)
                //     controller.debug_info();
            }
            if (dense_out){
                // compute output if t_eval is within this range
                outputter.output(cta, stepper, system_id, controller.t0, controller.hrun);
            }

            // copy from last t state to initial state
            // yn -> y0 // cta, T* dst, const T* src, const int N)
            cuda::detail::vector_copy(cta, stepper.y0, stepper.yn, stepper.system_size);
            cta.sync();
            // k7 (fn) -> k1 (f0), recompute anyway, no need to copy
            // cuda::detail::vector_copy(cta, stepper.k1, stepper.k7, stepper.system_size);
            // cta.sync();
            // increase t0 -> t0+h
            if(cta.thread_rank()==0)
                controller.t0_increment();
            cta.sync();
        } // end of t steps iteration
    } // end of events iteration
    // all done
}

template <class real_type, class ode_system_type, class event_type>
__global__ void solve_ivp_kernel(
    const int system_offset,
    const bool dense_out,
    ode_system_type ode,
    event_type  events,
    Stepper<real_type> * steppers,
    Controller<real_type> * controllers,
    DenseOutput<real_type> * outputs)
{
    // get the system index and patch index
    auto block_id = blockIdx.x;
    auto system_id = block_id + system_offset;
    auto cta = cg::this_thread_block();
    auto patch_id = cta.thread_rank();

    // get my system, one per block
    auto& stepper = steppers[block_id];
    auto& controller = controllers[block_id];
    auto& outputter = outputs[block_id];
    solve_device(cta, system_id, dense_out,
        ode,
        events,
        stepper,
        controller,
        outputter);
}

template <class real_type, class ode_system_type, class event_type>
void Solver<real_type, ode_system_type, event_type>::solve_ivp(const bool dense_out, const int systems, const int system_offset)
{
    int blocks = systems;

    solve_ivp_kernel<real_type, ode_system_type, event_type><<<blocks, threads>>>(
        system_offset, dense_out,
        ode,
        events,
        stepper_holder->steppers,
        controller_holder->controllers,
        output_holder->outputters);
    cudaCheckError("solve_ivp_kernel error");

    // all done, results saved in yeval
}

template <class real_type, class ode_system_type, class event_type>
void Solver<real_type, ode_system_type, event_type>::set_dense_output(const int neval, const real_type * tevals, real_type * yevals)
{
    output_holder = new output_holder_type(systems_batch, system_size, neval, tevals, yevals);
}

} // end of namespace cuda::ode

#endif
// end of file
