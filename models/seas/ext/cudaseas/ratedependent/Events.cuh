/*
 *  This is an example of an event required by the package
 *  An event defines a series of events, or changes to y(t), at t0, t1, t2, ... tn.
 *  (t0, tn) also used as starting and ending times for ode integration
 *  @note current implementation support changes at t0, not tn
 *  If there are no events, use FixedEvents structure from events.cuh,
 *         e.g., cuda::ode::dopri5::FixedEvents<T> tspan(t0, tn)
 */

#ifndef __rd_event_cuh__
#define __rd_event_cuh__

template <typename T>
struct __ALIGNED__ SEASEvents {
    // required event parameters
    int nevents; // number of events, including starting and ending time t0, tn
    const T* tevents;

    // other custom parameters
    int patches;
    int units;
    int systems;
    int system_size;
    int num_eq; // unique number of events
    int num_slips; // total number of events (including repeating ones)
    const T* ychange; // [systems, num_eq, patches * 2] - first all velocities in one direction, then the other
    const int* delta_tau_ix; // convert non-unique event_id to unique eq_id
    const int* delta_tau_ix_final; // same as before but for the final, spun-up period - can be switched to
    bool spun_up = false; // marker to decide with delta_tau_ix to use

    // debugging descriptor
    // this is run on cpu, so only a host function, it could print results as below (by making all arrays in managed memory)
    __host__ void describe() {
        cudaDeviceSynchronize(); // needed to sync data from gpu memory to cpu memory
        printf("SEASEvents\n");
        printf("patches = %i, units = %i, system_size = %i, systems = %i\n",
               patches, units, system_size, systems);
        printf("num_eq = %i, num_slips = %i, nevents = %i\n", num_eq, num_slips, nevents);
        printf("tevents = %g ... %g\n", tevents[0], tevents[nevents - 1]);
        printf("ychange = %g ... %g\n", ychange[0], ychange[systems * num_eq * patches * 2 - 1]);
        printf("delta_tau_ix = %i ... %i\n", delta_tau_ix[0], delta_tau_ix[num_slips - 1]);
        printf("delta_tau_ix_final = %i ... %i\n", delta_tau_ix_final[0], delta_tau_ix_final[num_slips - 1]);
    }

    // an example constructor
    SEASEvents (const int num_slips_, const int num_eq_, const T* tevents_, const T* delta_tau_div_alpha_h_,
                const int* delta_tau_ix_, const int* delta_tau_ix_final_, const int systems_, const int patches_, const int units_)
        : num_slips(num_slips_), num_eq(num_eq_), systems(systems_), patches(patches_), units(units_),
          tevents(tevents_), delta_tau_ix(delta_tau_ix_), delta_tau_ix_final(delta_tau_ix_final_),
          ychange(delta_tau_div_alpha_h_)
    {
        system_size = patches * units;
        nevents = num_slips + 2;
        // // this is run on cpu, so only a host function
        // describe();
    }

    // legacy constructor with a single delta_tau_ix
    SEASEvents (const int num_slips_, const int num_eq_, const T* tevents_, const T* delta_tau_div_alpha_h_,
                const int* delta_tau_ix_, const int systems_, const int patches_, const int units_)
        : num_slips(num_slips_), num_eq(num_eq_), systems(systems_), patches(patches_), units(units_),
          tevents(tevents_), delta_tau_ix(delta_tau_ix_), delta_tau_ix_final(delta_tau_ix_),
          ychange(delta_tau_div_alpha_h_)
    {
        system_size = patches * units;
        nevents = num_slips + 2;
        // // this is run on cpu, so only a host function
        // describe();
    }


    // keep this function
    __device__ const T* get_events_time(const int system_id)
    {
        // default, assuming all systems use the same event times
        return tevents;
    }

    // set spun-up marker
    __device__ void set_spun_up(bool new_status)
    {
        spun_up = new_status;
    }

    // default hook to be called by the ode solver
    __device__ void set_events_block(const cg::thread_block & cta,
        const int system_id, const int event_id, T* yn)
    {
        if (event_id == 0) return;
        assert (event_id < nevents - 1); // just to be sure

        // convert event_id (non-unique) to eq_id (unique)
        auto eq_id = (spun_up) ? delta_tau_ix_final[event_id - 1] : delta_tau_ix[event_id - 1];

        auto yevent = ychange + system_id * num_eq * patches * 2 + eq_id * patches * 2;
        // note one thread per patch, the iteration is for system_size > #total threads
        for (int id = cta.thread_rank(); id < patches * 2; id += cta.size())
            yn[id + patches * 2] += yevent[id]; // simply add changes, only for velocity

        // cta.sync();
        // return;
    };
};

#endif //__rd_event_cuh__
// end of file
