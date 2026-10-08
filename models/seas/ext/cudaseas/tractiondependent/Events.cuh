/*
 *  Traction-dependent events (identical to ratedependent)
 */

#ifndef __td_event_cuh__
#define __td_event_cuh__

#include "wright_omega.cuh"

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
    const T v_ratio_max; // maximum velocity [-] ratio relative to v_0, zero if no maximum
    const T* alpha_h; // [systems * patches]
    const T mu_over_2vs;
    const T v_0;
    const T rho;
    T v_max = v_ratio_max * v_0;
    T eta_v_max = mu_over_2vs * v_max;
    T rho_log_v_max_v_0 = rho + log(v_max / v_0);

    // constructor
    SEASEvents (const int num_slips_, const int num_eq_, const T* tevents_, const T* delta_tau_div_alpha_h_,
                const int* delta_tau_ix_, const int* delta_tau_ix_final_,
                const int systems_, const int patches_, const int units_, const T v_ratio_max_,
                const T* alpha_h_, const T mu_over_2vs_, const T v_0_, const T rho_)
        : num_slips(num_slips_), num_eq(num_eq_), systems(systems_), patches(patches_), units(units_),
          tevents(tevents_), delta_tau_ix(delta_tau_ix_), delta_tau_ix_final(delta_tau_ix_final_),
          ychange(delta_tau_div_alpha_h_), v_ratio_max(v_ratio_max_),
          alpha_h(alpha_h_), mu_over_2vs(mu_over_2vs_), v_0(v_0_), rho(rho_)
    {
        system_size = patches * units;
        nevents = num_slips + 2;
        cudaSafeCall(cudaMalloc(&spun_up, systems * sizeof(bool)));
        // the spin-up uses the final events too, not whatever the allocation held
        cudaSafeCall(cudaMemset(spun_up, true, systems * sizeof(bool)));
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
                // apply simple coseismic traction change
                auto tau1 = yn[id + patches * 2] + yevent[id];
                auto tau2 = yn[id + patches * 3] + yevent[id + patches];
                if (v_ratio_max > 0)
                {
                    // compute traction magnitude
                    auto tau_mag = sqrt(pow(tau1, 2) + pow(tau2, 2));
                    // compute velocity magnitude
                    auto minusc = mu_over_2vs / alpha_h[system_id * patches + id];
                    auto zprime = (
                        tau_mag / alpha_h[system_id * patches + id]
                        - rho + log(v_0 * minusc)
                    );
                    auto v_mag = wright_omega(zprime) / minusc;
                    // check magnitude
                    if (v_mag > v_max)
                    {
                        // direction of traction
                        auto tauhat_1 = tau1 / tau_mag;
                        auto tauhat_2 = tau2 / tau_mag;
                        // get traction but with v_max instead
                        auto tau_mag_new = rho_log_v_max_v_0 * alpha_h[system_id * patches + id] + eta_v_max;
                        // get traction vector
                        tau1 = tauhat_1 * tau_mag_new;
                        tau2 = tauhat_2 * tau_mag_new;
                    }
                }
                yn[id + patches * 2] = tau1;
                yn[id + patches * 3] = tau2;
            }
        }
    };
};

#endif //__td_event_cuh__
