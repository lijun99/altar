// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2022 - 2023 california institute of technology
// all rights reserved

/*
 *  This is an example of an event required by the package
 *  An event defines a series of events, or changes to y(t), at t0, t1, t2, ... tn.
 *  (t0, tn) also used as starting and ending times for ode integration
 *  @note current implementation support changes at t0, not tn
 *  If there are no events, use FixedEvents structure from events.cuh,
 *         e.g., cuda::ode::dopri5::FixedEvents<T> tspan(t0, tn)
 */

#ifndef altar_models_seas_cuda_linearviscous_events_cuh
#define altar_models_seas_cuda_linearviscous_events_cuh

// enclosed in a name space
namespace altar::models::seas::cuda::linearviscous {

template <typename T>
struct __ALIGNED__ Events {
    // required event parameters
    int nevents; // number of coseismic changes, including starting and ending time t0, tn
    const T* tevents; // [nevents], assume the same for all systems here, otherwise, rewrite

    // other custom parameters
    int system_size;
    const T * yevents; // [nevents-1, system_size], assuming nothing happens at tn, and same for all systems

    // an example constructor
    Events (const int nevents_, const T* tevents_, const T* yevents_, const int system_size_)
        : nevents(nevents_), system_size(system_size_), tevents(tevents_), yevents(yevents_)
    {
    };

    // keep this function
    __device__ const T* get_events_time(const int system_id)
    {
        // default, assuming all systems use the same event times
        return tevents;
    };

    // compatibility with ratedependent object
    __device__ void set_spun_up(bool new_status)
    {
        return;
    }

    // default hook to be called by the ode solver
    // here, we simply add the changes (yevents) to y(tevent)
    __device__ void set_events_block(const cg::thread_block & cta,
        const int system_id, const int event_id, T* yn)
    {
        auto yevent = yevents + event_id*system_size;
        // note one thread per patch, the iteration is for system_size > #total threads
        for(int id = cta.thread_rank(); id<system_size; id+=cta.size())
            yn[id] += yevent[id]; // simply add coseismic changes
        // cta.sync();
        // return;
    };
};

} // end of namespace

#endif // altar_models_seas_cuda_linearviscous_events_cuh
// end of file
