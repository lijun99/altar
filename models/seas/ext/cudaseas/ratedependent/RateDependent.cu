// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2023 california institute of technology
// all rights reserved
//

// for the build system
#include <portinfo>

// get my class declaration
#include "RateDependent.h"

namespace altar::models::seas::cuda::ratedependent {

// Initialize model parameters
// suffix underline indicate class parameters
template <typename T>
void RateDependent<T>::initialize(
        int num_systems_,
        int systems_batch_,
        int max_cycles_,
        int num_t_eval_,
        T* t_eval_joint_sec_,
        int num_ix_eq_,
        int num_eq_,
        int* ix_eq_joint_,
        T* t_events_,
        T v_0_,
        T mu_over_2vs_,
        int num_inner_patches_,
        T* K_inner_inner_onfault_,
        T* K_inner_asperities_v_plate_,
        T* v_plate_ddcs_proj_eff_inner_,
        T* v_init_,
        T atol_,
        T rtol_,
        T spinup_atol_,
        T spinup_rtol_)
{
    // general variables
    num_systems = num_systems_;
    systems_batch = systems_batch_;

    // cycles
    max_cycles = max_cycles_;
    num_t_eval = num_t_eval_;
    t_eval_joint_sec = t_eval_joint_sec_;

    // events
    num_ix_eq = num_ix_eq_;
    n_events = num_ix_eq + 2;
    num_eq = num_eq_;
    ix_eq_joint = ix_eq_joint_;
    t_events = t_events_;

    // rheology
    v_0 = v_0_;
    mu_over_2vs = mu_over_2vs_;

    // fault
    num_inner_patches = num_inner_patches_;
    system_size = num_inner_patches * UNITS;
    K_inner_inner_onfault = K_inner_inner_onfault_;
    K_inner_asperities_v_plate = K_inner_asperities_v_plate_;
    v_plate_ddcs_proj_eff_inner = v_plate_ddcs_proj_eff_inner_;
    v_init = v_init_;

    // ode
    atol = atol_;
    rtol = rtol_;
    spinup_atol = spinup_atol_;
    spinup_rtol = spinup_rtol_;
    conv_i_start = (UNITS / 2) * num_inner_patches;
    conv_i_stop = UNITS * num_inner_patches - 1;

    // output variable
    cudaMallocManaged(&sim_state, num_systems * num_t_eval * system_size * sizeof(T));
}

template <typename T> 
void RateDependent<T>::set_system_odes(
    T* alpha_h_vec_,
    T* delta_tau_bounded_
    )
{
    // save system-specific rheology and event realization
    alpha_h_vec = alpha_h_vec_;
    delta_tau_bounded = delta_tau_bounded_;

    // create an instance of odefunc
    OdeType odefunc {num_inner_patches, UNITS, num_systems, alpha_h_vec, mu_over_2vs, v_0, K_inner_inner_onfault,
                     K_inner_asperities_v_plate, v_plate_ddcs_proj_eff_inner};

    // create an instance of events (including starting/ending time)
    EventType events {n_events, t_events, delta_tau_bounded, alpha_h_vec, num_systems, num_inner_patches, UNITS};

    // create the solver
    SolverType solver {odefunc, events, atol, rtol, spinup_atol, spinup_rtol, systems_batch};
    solver.set_dense_output(num_t_eval, t_eval_joint_sec, sim_state);
}

template <typename T> 
void RateDependent<T>::forward_model_batch () {
    for (int system_offset = 0; system_offset < num_systems; system_offset += systems_batch)
    {
        // check how many systems are left
        auto systems_to_process = min(systems_batch, num_systems - system_offset);
         // set initial values
        solver->set_init_values(v_init, USE_V_INIT_FOR_ALL, systems_to_process, system_offset);
        // call the solver
        solver->solve_ivp_cycles(DENSE_OUT, systems_to_process, system_offset,
                                 conv_i_start, conv_i_stop, max_cycles);
        cudaDeviceSynchronize();
    }
}

// explicit instantiation
template class altar::models::seas::cuda::ratedependent::RateDependent<float>;
template class altar::models::seas::cuda::ratedependent::RateDependent<double>;

} // end of namespace
// end of file
