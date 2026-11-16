// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2022 - 2023 california institute of technology
// all rights reserved

/**
 * CUDA package to solve initial value problem of ODE systems
 * It solves a batch of ode systems at the same time, with each thread block for each system
 * Currently implemented with
 *     Runge-Kutta Dormand Prince 54 ODE solver
 **/

// code guard
#ifndef __cuda_ode_cuh__
#define __cuda_ode_cuh__

// Solve normal ODE problems within (t0, tn)
#include <dopri5/solver.cuh>
// Solve ODEs within (t0, tn) for many cycles, targeting the spinup/loading process
#include <dopri5/spinup_solver.cuh>

#endif // __cuda_ode_cuh__
// end of file
