// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2022 - 2026 california institute of technology
// all rights reserved

/**
 * method.cuh
 * the integration method a {Solver} uses: its stepper, error controller, and dense output
 *
 * a method provides, for each system (one thread block),
 *   stepper:    y0, yn, system_size; integrate(cta, system_id, t0, h, ode),
 *               set_init_value(cta, t0, y0, system_id, ode), set_f0_value(cta, t0, system_id, ode)
 *   controller: t0, hnext, hrun, converged, t1reached; init_run(t0, t1), check_reach_t1(),
 *               t0_increment(), select_initial_step(cta, system_id, stepper, ode),
 *               check_convergence(cta, stepper), record_step(cta, stepper),
 *               reset_statistics(), and the step statistics in {StepStatistics}
 *   output:     reset(cta), output(cta, stepper, system_id, t0, h)
 * and their holders, which allocate one of each per system:
 *   StepperHolder(ode, systems), ControllerHolder(systems, atol, rtol),
 *   DenseOutputHolder(systems, system_size, neval, tevals, yevals)
 **/

// code guard
#ifndef cuda_ode_dopri5_method_cuh
#define cuda_ode_dopri5_method_cuh

#include "stepper.cuh"
#include "controller.cuh"
#include "dense_output.cuh"

namespace cuda::ode::dopri5 {

// the step statistics of one system, over one solve
struct StepStatistics {
    int accepted; // accepted steps
    int rejected; // rejected steps
    int stiff; // accepted steps whose size was limited by stability rather than accuracy
    int cycles; // spin-up cycles, zero without spin up
    bool failed; // whether the integration was given up on
    double hmin; // the smallest accepted step
    double hmax; // the largest accepted step
};

// the Runge-Kutta Dormand Prince 5(4) method
template <class T>
struct Dopri5 {
    using stepper_type = Stepper<T>;
    using stepper_holder_type = StepperHolder<T>;
    using controller_type = Controller<T>;
    using controller_holder_type = ControllerHolder<T>;
    using output_type = DenseOutput<T>;
    using output_holder_type = DenseOutputHolder<T>;
};

} // end of namespace cuda::ode::dopri5

#endif // cuda_ode_dopri5_method_cuh
// end of file
