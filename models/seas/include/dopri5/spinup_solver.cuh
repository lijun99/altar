// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2022 - 2023 california institute of technology
// all rights reserved

/**
 * iterative_solver.cuh
 * A custom ode solver for spin-up, iteratively solve an ode system until the initial and final values are the same
 **/

// code guard
#ifndef cuda_ode_dopri5_spinup_solver_cuh
#define cuda_ode_dopri5_spinup_solver_cuh

#include "solver.cuh"
#include "spinup_controller.cuh"


// enclosed in namespace
namespace cuda::ode::dopri5 {

// declaration of the dopri5 solver class`
template <class real_type, class ode_system_type, class event_type = FixedEvents<real_type>>
struct SpinupSolver : public Solver<real_type, ode_system_type, event_type>
{

    using single_solver_type = Solver<real_type, ode_system_type, event_type>;
    using spinup_controller_type = SpinupController<real_type>;
    using spinup_controller_holder_type = SpinupControllerHolder<real_type>;

    int max_iterations;
    spinup_controller_holder_type * spinup_controller_holder;

    // constructor
    // @param [atol, rtol] error control for RK stepper
    // @param [spinup_atol, spinup_rtol] error control for spinup convergence
    SpinupSolver(ode_system_type & ode_, event_type & events_,
        const real_type atol_=1e-8, const real_type rtol_=1e-6,
        const real_type spinup_atol_ = 1e-6, const real_type spinup_rtol_ = 1e-3,
        const int systems_batch_=8192)
        : single_solver_type(ode_, events_, atol_, rtol_, systems_batch_)
    {
        spinup_controller_holder = new spinup_controller_holder_type(this->systems_batch, this->system_size,
            spinup_atol_, spinup_rtol_);
    };

    void solve_ivp_cycles(const bool dense_out, const int systems, const int system_offset, const int max_cycles);

    void solve_ivp_cycles(const bool dense_out, const int systems, const int system_offset,
        const int index_start, const int index_end, const int max_cycles);

};

// re-set f(t0, y0) at the beginning of each cycle
template <class real_type, class ode_system_type, class event_type>
__device__ void reset_f0_value( const cg::thread_block & cta,
    const int system_id,
    ode_system_type & ode,
    event_type &  events,
    Stepper<real_type> & stepper
    )
{
    auto tevents = events.tevents;
    auto t0 = tevents[0];
    // re-compute f0 = f(t0, y0) due to the possible dependence on t0
    // assume y0 has already been copied or set
    stepper.set_f0_value(cta, t0, system_id, ode);
    cta.sync();
}

template <class real_type, class ode_system_type, class event_type>
__global__ void solve_ivp_cycles_kernel(const int system_offset,
    const bool dense_out,
    ode_system_type ode,
    event_type  events,
    Stepper<real_type> * steppers,
    Controller<real_type> * controllers,
    SpinupController<real_type> * spinup_controllers,
    DenseOutput<real_type> * outputs,
    const int index_start, const int index_end,
    const int max_cycles)
{
    // get the system index and patch index
    auto block_id = blockIdx.x;
    auto system_id = block_id + system_offset;
    auto cta = cg::this_thread_block();

    // get my system, one per block
    auto& stepper = steppers[block_id];
    auto& controller = controllers[block_id];
    auto& outputter = outputs[block_id];
    auto& spinup_controller = spinup_controllers[block_id];

    // disable dense_out for spin up iterations
    bool dense_out_run = false;
    // record y0 as an initial value for yn for convergence check
    spinup_controller.record(cta, stepper.y0, index_start, index_end);

    // repeat cycles until convergence or max_cycles reached
    bool converged = false;
    int icycle = 0;
    while (!converged && icycle<max_cycles)
    {
        // make another cycle
        reset_f0_value(cta, system_id, ode, events, stepper);
        solve_device(cta, system_id, dense_out_run,
            ode,
            events,
            stepper,
            controller,
            outputter);
        cta.sync();

        // check for convergence
        converged = spinup_controller.check_convergence2(cta, stepper.yn, index_start, index_end);
        icycle++;
        // keep record of the final y(tn) for next convergence check
        spinup_controller.record(cta, stepper.yn, index_start, index_end);
        cta.sync();
        if (cta.thread_rank() == 0)
            printf("Cycle %i completed\n", icycle);
    }
    //cta.sync();

    // // if debugging cycles
    // if(cta.thread_rank()==0)
    //     printf("spinup cycles finished in %d steps, convergence is %s\n", icycle, converged ? "true" : "false");

    // TODO add warning output if not converged
    if ((cta.thread_rank() == 0) && !converged)
        printf("WARNING: spinup_solver did not converge\n");

    //converged, last run for dense_out
    if(dense_out)
    {
        reset_f0_value(cta, system_id, ode, events, stepper);
        solve_device(cta, system_id, dense_out,
            ode,
            events,
            stepper,
            controller,
            outputter);
    }
    // all done
}

template <class real_type, class ode_system_type, class event_type>
void SpinupSolver<real_type, ode_system_type, event_type>::solve_ivp_cycles(
    const bool dense_out, const int systems, const int system_offset,
    const int index_start, const int index_end, // start end end indices of yn for convergence check
    const int max_cycles)
{
    auto patches = this->ode.patches;
    int threads;
    if(patches <= 32)
        threads = 32;
    else if (patches <= 64)
        threads = 64;
    else if (patches <= 128)
        threads =128;
    else if (patches <= 256)
        threads = 256;
    else if (patches <=512 )
        threads = 512;
    else
        threads = 1024;

    int blocks = systems;

    // printf("    inside spinup_solver.cuh:solve_ivp_cycles (patches=%i, threads=%i, blocks=%i)\n",
    //        patches, threads, blocks);

    solve_ivp_cycles_kernel<real_type, ode_system_type, event_type><<<blocks, threads>>>(
        system_offset,
        dense_out,
        this->ode,
        this->events,
        this->stepper_holder->steppers,
        this->controller_holder->controllers,
        spinup_controller_holder->controllers,
        this->output_holder->outputters,
        index_start, index_end,
        max_cycles);
    cudaCheckError("solve_ivp_kernel error");
    // all done, return the yevals
}

template <class real_type, class ode_system_type, class event_type>
void SpinupSolver<real_type, ode_system_type, event_type>::solve_ivp_cycles(const bool dense_out, const int systems, const int system_offset, const int max_cycles)
{
    solve_ivp_cycles(dense_out, systems, system_offset,
        0, this->system_size-1, max_cycles);
}

} // end of namespace cuda::ode::dopri5

#endif
// end of file
