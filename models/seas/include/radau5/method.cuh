// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2026 california institute of technology
// all rights reserved

/**
 * method.cuh
 * the Radau IIA (order 5) method, for the solvers in dopri5/solver.cuh; see dopri5/method.cuh
 **/

// code guard
#ifndef cuda_ode_radau5_method_cuh
#define cuda_ode_radau5_method_cuh

#include "stepper.cuh"
#include "controller.cuh"
#include "dense_output.cuh"

namespace cuda::ode::radau5 {

template <class T>
struct Radau5 {
    using stepper_type = Stepper<T>;
    using stepper_holder_type = StepperHolder<T>;
    using controller_type = Controller<T>;
    using controller_holder_type = ControllerHolder<T>;
    using output_type = DenseOutput<T>;
    using output_holder_type = DenseOutputHolder<T>;
};

} // end of namespace cuda::ode::radau5

#endif // cuda_ode_radau5_method_cuh
// end of file
