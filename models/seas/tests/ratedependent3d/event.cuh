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
    T* tevents;

    // other custom parameters
    int patches;
    int units;
    int systems;
    int system_size;
    int num_eq;
    T * ychange; // [systems, num_eq=nevents-2, patches * 2] - first all velocities in one direction, then the other

    // index function for delta_tau_bounded, 3D
    int i_dtau (int i0, int i1, int i2) {
        assert ((i0 < num_eq) && (i1 < patches) && (i2 < 2));
        return (i2) + (i1 * 2) + (i0 * 2 * patches);
    }

    // an example constructor
    SEASEvents (const int nevents_, const T* tevents_, const T* delta_tau, const T* alpha_h_vec, const int systems_, const int patches_, const int units_)
        : nevents(nevents_), systems(systems_), patches(patches_), units(units_)
    {
        system_size = patches * units;
        num_eq = nevents - 2;

        // allocate the event time
        cudaMallocManaged(&tevents, nevents * sizeof(T));
        cudaMemcpy(tevents, tevents_, nevents * sizeof(T), cudaMemcpyDefault);
        // for (auto i = 0; i < nevents; i++)
        //     tevents[i] = tevents_[i];

        // set up the event changes
        // ychange has first all the patches for unit 1, then unit 2, etc., so need to change order
        cudaMallocManaged(&ychange, systems * num_eq * patches * 2 * sizeof(T));
        int iyoff;
        int ialphaoff;
        for (auto isys = 0; isys < systems; isys++) {
            iyoff = isys * num_eq * patches * 2;
            ialphaoff = isys * patches;
            for (auto ievent = 0; ievent < num_eq; ievent++) {
                for (auto i = 0; i < patches; i++) {
                    ychange[iyoff + ievent * patches * 2 + i] =
                        (T) delta_tau[iyoff + i_dtau(ievent, i, 0)] / alpha_h_vec[ialphaoff + i];
                    ychange[iyoff + ievent * patches * 2 + patches + i] =
                        (T) delta_tau[iyoff + i_dtau(ievent, i, 1)] / alpha_h_vec[ialphaoff + i];
                }
            }
        }
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
        if (event_id == 0) return;

        auto yevent = ychange + (event_id - 1) * patches * 2 + system_id * num_eq * patches * 2;
        // note one thread per patch, the iteration is for system_size > #total threads
        for(int id = cta.thread_rank(); id < patches * 2; id += cta.size())
            yn[id + patches * 2] += yevent[id]; // simply add changes, only for velocity

        cta.sync();
        return;
    };
};

#endif //__rd_event_cuh__
// end of file
