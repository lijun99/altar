/*
 *  Traction-dependent events (identical to ratedependent)
 */

#ifndef __td_event_cuh__
#define __td_event_cuh__

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
    const T* ychange; // [systems, num_eq, patches * 2]
    const int* delta_tau_ix;
    const int* delta_tau_ix_final;
    bool* spun_up;
    const T v_ratio_max; // maximum relative velocity [-], zero if no maximum

    // constructor
    SEASEvents (const int num_slips_, const int num_eq_, const T* tevents_, const T* delta_tau_div_alpha_h_,
                const int* delta_tau_ix_, const int* delta_tau_ix_final_,
                const int systems_, const int patches_, const int units_, const T v_ratio_max_)
        : num_slips(num_slips_), num_eq(num_eq_), systems(systems_), patches(patches_), units(units_),
          tevents(tevents_), delta_tau_ix(delta_tau_ix_), delta_tau_ix_final(delta_tau_ix_final_),
          ychange(delta_tau_div_alpha_h_), v_ratio_max(v_ratio_max_)
    {
        system_size = patches * units;
        nevents = num_slips + 2;
        cudaSafeCall(cudaMalloc(&spun_up, systems * sizeof(bool)));
    }

    void deallocate()
    {
        if(spun_up != nullptr)
            cudaSafeCall(cudaFree(spun_up));
    }

    ~SEASEvents() noexcept(false) {}

    __device__ const T* get_events_time(const int system_id)
    {
        return tevents;
    }

    __device__ void set_spun_up(const int system_id, const bool new_status)
    {
        spun_up[system_id] = new_status;
    }

    __device__ void set_events_block(const cg::thread_block & cta,
        const int system_id, const int event_id, T* yn)
    {
        if (event_id == 0)
        {
            for (int id = cta.thread_rank(); id < patches; id += cta.size())
            {
                yn[id] = 0;
                yn[id + patches] = 0;
            }
        }
        else
        {
            assert (event_id < nevents - 1);

            auto eq_id = (spun_up[system_id]) ? delta_tau_ix_final[event_id - 1] : delta_tau_ix[event_id - 1];

            auto yevent = ychange + system_id * num_eq * patches * 2 + eq_id * patches * 2;
            for (int id = cta.thread_rank(); id < patches; id += cta.size())
            {
                auto zeta1 = yn[id + patches * 2] + yevent[id];
                auto zeta2 = yn[id + patches * 3] + yevent[id + patches];
                if (v_ratio_max > 0)
                {
                    assert (false); // not updated to work with traction yet
                    auto vr1 = exp(zeta1);
                    auto vr2 = exp(zeta2);
                    auto vrmag = sqrt(vr1 * vr1 + vr2 * vr2);
                    if (vrmag > v_ratio_max)
                    {
                        auto ratio = min(vrmag, v_ratio_max) / vrmag;
                        vr1 *= ratio;
                        vr2 *= ratio;
                        zeta1 = log(vr1);
                        zeta2 = log(vr2);
                    }
                }
                yn[id + patches * 2] = zeta1;
                yn[id + patches * 3] = zeta2;
            }
        }
    };
};

#endif //__td_event_cuh__
