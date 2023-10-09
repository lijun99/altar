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
#include <iostream>

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
        int* delta_tau_bounded_indices_,
        int* ix_eq_joint_,
        T* t_events_,
        T v_0_,
        T mu_over_2vs_,
        int num_inner_patches_,
        T* K_inner_inner_onfault_,
        T* K_inner_asperities_v_plate_,
        T* v_plate_ddcs_proj_eff_inner_,
        T* v_init_,
        T* sim_state_,
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
    num_eq = num_eq_;
    ix_eq_joint = ix_eq_joint_;
    delta_tau_bounded_indices = delta_tau_bounded_indices_;
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
    sim_state = sim_state_;
}

template <typename T>
void RateDependent<T>::set_system_odes(
    T* alpha_h_vec_,
    T* delta_tau_div_alpha_h_
    )
{
    printf("inside RateDependent.cu:set_system_odes\n");
    // save system-specific rheology and event realization
    alpha_h_vec = alpha_h_vec_;
    delta_tau_div_alpha_h = delta_tau_div_alpha_h_;

    printf("  assigned pointers\n");

    // create an instance of odefunc
    odefunc = new OdeType{num_inner_patches, UNITS, num_systems, alpha_h_vec, mu_over_2vs, v_0, K_inner_inner_onfault,
                     K_inner_asperities_v_plate, v_plate_ddcs_proj_eff_inner};

    printf("  initialized odefunc\n");

    // create an instance of events (including starting/ending time)
    events = new EventType{num_ix_eq, num_eq, t_events, delta_tau_div_alpha_h, delta_tau_bounded_indices,
                      num_systems, num_inner_patches, UNITS};

    printf("  initialized events\n");

    // create the solver
    solver = new SolverType{*odefunc, *events, atol, rtol, spinup_atol, spinup_rtol, systems_batch};

    printf("  initialized solver\n");
    solver->set_dense_output(num_t_eval, t_eval_joint_sec, sim_state);

    printf("  set dense output\n");
}

template <typename T>
void RateDependent<T>::forward_model_batch () {
    printf("inside RateDependent.cu:forward_model_batch\n");

    std::cout << "Debug forward_model_batch "
        << "num_systems " << num_systems
        << "systems_batch " << systems_batch
        << "system_size " << system_size
        << std::endl;

    for (int system_offset = 0; system_offset < num_systems; system_offset += systems_batch)
    {
        // check how many systems are left
        auto systems_to_process = min(systems_batch, num_systems - system_offset);
        printf("  processing systems %i to %i\n", system_offset, system_offset + systems_to_process - 1);
         // set initial values
        solver->set_init_values(v_init, USE_V_INIT_FOR_ALL, systems_to_process, system_offset);
        printf("  set initial values done \n");
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
