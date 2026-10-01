/*
 *  This is an example of an event required by the package
 *  An event defines a series of events, or changes to y(t), at t0, t1, t2, ... tn.
 *  (t0, tn) also used as starting and ending times for ode integration
 *  @note current implementation support changes at t0, not tn
 *  If there are no events, use FixedEvents structure from events.cuh,
 *         e.g., cuda::ode::dopri5::FixedEvents<T> tspan(t0, tn)
 */

#ifndef __mytestevent_cuh__
#define __mytestevent_cuh__

template <typename T>
struct __ALIGNED__ MyTestEvents {
    // required event parameters
    int nevents; // number of events, including starting and ending time t0, tn
    T* tevents;

    // other custom parameters
    int system_size;
    T * ychange; // [nevents-1, system_size]

    // an example constructor
    MyTestEvents (const int nevents_, const T t0, const T tn, const int system_size_)
        : nevents(nevents_), system_size(system_size_)
    {
        // set up the event time
        cudaMallocManaged(&tevents, nevents*sizeof(T));
        auto tstep = (tn-t0)/(nevents-1);
        for(auto i=0; i<nevents; i++)
            tevents[i] = t0 + tstep*i;

        // set up the event changes
        cudaMallocManaged(&ychange, (nevents-1)*system_size*sizeof(T));
        for(auto ievent = 0; ievent<nevents-1; ievent++)
            for(auto i=0; i<system_size; i++)
                ychange[ievent*system_size+i] = (T)1.0;
    }

    // keep this function
    __device__ const T* get_events_time(const int system_id)
    {
        // default, assuming all systems use the same event times
        return tevents;
    }

    // default hook to be called by the ode solver
    __device__ void set_events_block(const cg::thread_block & cta,
        const int system_id, const int event_id, T* yn)
    {
        auto yevent = ychange + event_id*system_size;
        // note one thread per patch, the iteration is for system_size > #total threads
        for(int id = cta.thread_rank(); id<system_size; id+=cta.size())
            yn[id] += yevent[id]; // simply add changes
        return;
    };
};

#endif //__mytestevent_cuh__
// end of file