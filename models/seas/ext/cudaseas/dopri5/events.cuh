// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2022-2023 california institute of technology
// all rights reserved

/**
 * events.cuh
 * a template to define ODE times and events
 * when following the template, please keep the naming convention for member variables and functions
 * Fixed Events - a simple event with starting and ending time (t0, t1)
 **/

// code guard
#ifndef cuda_ode_dopri5_events_cuh
#define cuda_ode_dopri5_events_cuh

#include "external.h"

namespace cuda::ode::dopri5 {

// a base class for events
// here assume (t0, t1), or (t0, t1, t2, ... t_{n-1})
template <typename T>
struct FixedEvents {

    int nevents;  // number of events
    T* tevents;   // pointer to time of events, accessible by cuda kernel

    // set the event times from a preset
    __host__ void init(const int n_, T* t_)
    {
        nevents = n_;
        tevents = t_;
    }

    // set the event by starting and ending time
    __host__ void init(T t0=0, T t1=1)
    {
        nevents = 2;
        cudaSafeCall(cudaMallocManaged((void **)&tevents, nevents*sizeof(T)));
        tevents[0] = t0;
        tevents[1] = t1;
    }

    // provide the event time information
    // may be customized, e.g., to generate random time points
    __device__ const T* get_events_time(const int system_id)
    {
        // default, assuming all systems use the same event times
        return tevents;
    }

    // provide the generated/saved event time
    __device__ const T* get_current_events_time(const int system_id)
    {
        // default, assuming all systems use the same event times
        return tevents;
    }

    __device__ void set_events(const int system_id, const int patch_id, T* yn)
    {
        // default, do nothing
        return;
    }

    // default hook to be called by the ode solver to set changes to y at event_id
    __device__ void set_events_block(const cg::thread_block & cta,
        const int system_id, const int event_id, T* yn)
    {
        // default, do nothing
        return;
    };

    // constructor
    FixedEvents(T t0=0, T t1=1)
    {
        init(t0, t1);
    }

};


} // end of namespace cuda::ode::dopri5

#endif // cuda_ode_dopri5_events_cuh
// end of file
