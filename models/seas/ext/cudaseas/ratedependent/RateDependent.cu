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

// get displacement routines
#include "Displacement.cuh"

#include <iostream>

namespace altar::models::seas::cuda::ratedependent {

// Initialize model parameters
// suffix underline indicate class parameters
template <typename T>
void RateDependent<T>::initialize(
        int num_systems_,
        int systems_batch_,
        int max_cycles_,
        int num_t_obs_,
        T* t_obs_sec_,
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
        T* state_init_,
        T* sim_state_,
        T atol_,
        T rtol_,
        T spinup_atol_,
        T spinup_rtol_,
        int num_stations_)
{
    // general variables
    num_systems = num_systems_;
    systems_batch = systems_batch_;

    // cycles
    max_cycles = max_cycles_;
    num_t_obs = num_t_obs_;
    t_obs_sec = t_obs_sec_;

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
    state_init = state_init_;

    // ode
    atol = atol_;
    rtol = rtol_;
    spinup_atol = spinup_atol_;
    spinup_rtol = spinup_rtol_;
    conv_i_start = (UNITS / 2) * num_inner_patches;
    conv_i_stop = UNITS * num_inner_patches - 1;
    sim_state = sim_state_;

    // observers
    num_stations =  num_stations_;
}

template <typename T>
void RateDependent<T>::forward_model_batch(
    const T* alpha_h_vec, // (a-b)*sigma_E strength parameter on fault patches (num_systems, num_inner_patches, ) [Pa]
    const T* delta_tau_div_alpha_h, // stress change for each system and earthquake divided by alpha_h (num_systems, num_eq, num_inner_patches, 2) [-]
    const T* G_surf, // Displacement kernel for all stations (1, 2*num_inner_patches, 3*num_stations) [-]
    T* obs_disp,  // Surface observations for all stations (num_systems, num_t_obs, 3*num_stations) [m]
    const int num_systems // batch size <=samples (in AlTar, not all samples are computed in simulations)
) {
    // create an instance of odefunc
    odefunc = new OdeType{num_inner_patches, UNITS, num_systems, alpha_h_vec, mu_over_2vs, v_0, K_inner_inner_onfault,
                          K_inner_asperities_v_plate, v_plate_ddcs_proj_eff_inner};

    // create an instance of events (including starting/ending time)
    events = new EventType{num_ix_eq, num_eq, t_events, delta_tau_div_alpha_h, delta_tau_bounded_indices,
                           num_systems, num_inner_patches, UNITS};

    // create the solver
    solver = new SolverType{*odefunc, *events, atol, rtol, spinup_atol, spinup_rtol, systems_batch};
    solver->set_dense_output(num_t_obs, t_obs_sec, sim_state);

    for (int system_offset = 0; system_offset < num_systems; system_offset += systems_batch)
    {
        // check how many systems are left
        auto systems_to_process = min(systems_batch, num_systems - system_offset);
        // printf("  processing systems %i to %i\n", system_offset, system_offset + systems_to_process - 1);
        // set initial values
        solver->set_init_values(state_init, USE_STATE_INIT_FOR_ALL, systems_to_process, system_offset);
        // call the solver
        solver->solve_ivp_cycles(DENSE_OUT, systems_to_process, system_offset,
                                 conv_i_start, conv_i_stop, max_cycles);
        // cudaDeviceSynchronize();
    }

    // save the t=0 sim_state for the first sample to state_int, to be used the init
    // state_int [2(slip/stress), 2(slip_components), patches]
    // sim_state [systems, t_steps, 2(slip/stress), 2(slip_components), patches]
    // only save slip rate (stress)
    auto state_init_copy_start = state_init + 2*num_inner_patches;
    auto sim_state_copy_start = sim_state + 2*num_inner_patches;
    cudaSafeCall(cudaMemcpy(state_init_copy_start, sim_state_copy_start,
        2*num_inner_patches*sizeof(T), cudaMemcpyDeviceToDevice));

    // cudaDeviceSynchronize();
    // for(auto i=0; i<4*num_inner_patches; i++)
    //    std::cout << i<< " "
    //        << state_init[i] << " "
    //        << sim_state[i] << " "
    //        << sim_state[num_t_obs*4*num_inner_patches+i] << "\n";

    // convert logairthmic velocity to linear one
    convert_slip_rate<T>(sim_state, num_systems, num_t_obs, num_inner_patches, v_0);

    // call displacement routines - see details in Displacement.cuh for different implementations
    // assume Cd is a constant and gf is time independent
    compute_displacement_impl1(
        obs_disp, sim_state, G_surf, num_systems, num_t_obs, num_inner_patches, 3 * num_stations, v_0,
        (T) 1.0, (T) 0.0); // alpha beta for gemm C = alpha A B + beta C

    // uncomment following to use the displacement subtraction from t_num_eq

    if(num_t_eq > 0)
        subtract_displacement_from_teq(
            obs_disp, num_systems, num_t_obs, 3 * num_stations,
            t_eq_indices, num_t_eq);


}

// explicit instantiation
template class altar::models::seas::cuda::ratedependent::RateDependent<float>;
template class altar::models::seas::cuda::ratedependent::RateDependent<double>;

} // end of namespace
// end of file
