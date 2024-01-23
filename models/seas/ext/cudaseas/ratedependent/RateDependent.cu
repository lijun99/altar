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
        int cuda_batch_size_,
        int max_cycles_,
        int num_t_obs_,
        T* t_obs_sec_,
        int num_ix_eq_,
        int num_eq_,
        int* delta_tau_bounded_indices_,
        T* t_events_,
        int* i_slips_obs_,
        int n_slips_obs_,
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
    cuda_batch_size  = cuda_batch_size_;

    // cycles
    max_cycles = max_cycles_;
    num_t_obs = num_t_obs_;
    t_obs_sec = t_obs_sec_;

    // events
    num_ix_eq = num_ix_eq_;
    num_eq = num_eq_;
    delta_tau_bounded_indices = delta_tau_bounded_indices_;
    t_events = t_events_;
    i_slips_obs = i_slips_obs_;
    n_slips_obs = n_slips_obs_;

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
    const T* alpha_h_vec, // (a-b)*sigma_E strength parameter on fault patches (num_forward_batch, num_inner_patches, ) [Pa]
    const T* delta_tau_div_alpha_h, // stress change for each system and earthquake divided by alpha_h (num_forward_batch, num_eq, num_inner_patches * 2) [-]
    const T* G_surf, // Displacement kernel for all stations (1, 2*num_inner_patches, 3*num_stations) [-]
    T* obs_disp,  // Surface observations for all stations (num_forward_batch, num_t_obs, 3*num_stations) [m]
    const int num_forward_batch, // forward model batch(system) size <= cuda_batch_size (in AlTar, not all samples are computed in simulations)
    const int num_threads, // number of threads 1 <= num_threads <= 1024, 0 means internally estimated
    bool verbose = false // whether to print info and progress indicators or not
) {

    // make sure the total number of systems to process is smaller than the max batch size by cuda
    assert(num_forward_batch <= cuda_batch_size);

    // create an instance of odefunc
    odefunc = new OdeType{num_inner_patches, UNITS, num_forward_batch, alpha_h_vec, mu_over_2vs, v_0, K_inner_inner_onfault,
                          K_inner_asperities_v_plate, v_plate_ddcs_proj_eff_inner};

    // create an instance of events (including starting/ending time)
    events = new EventType{num_ix_eq, num_eq, t_events, delta_tau_div_alpha_h, delta_tau_bounded_indices,
                           num_forward_batch, num_inner_patches, UNITS};

    // create the solver
    solver = new SolverType{*odefunc, *events, atol, rtol, spinup_atol, spinup_rtol, num_forward_batch, num_threads};
    solver->set_dense_output(num_t_obs, t_obs_sec, sim_state);

    int system_offset = 0;
    auto systems_to_process = num_forward_batch;

    // move this to python
    // if (verbose)
    //     printf("Processing systems %i to %i\n", system_offset, system_offset + systems_to_process - 1);

    // set initial values, false means each system gets an own copy
    solver->set_init_values(state_init, false, systems_to_process, system_offset);
    // call the solver
    solver->solve_ivp_cycles(DENSE_OUT, systems_to_process, system_offset,
                             conv_i_start, conv_i_stop, max_cycles, verbose);
    cudaDeviceSynchronize();


    // save the t=0 sim_state for the first sample to state_int, to be used the init
    // @NOTE there should not be an event update at t=0. Otherwise, make sure last t_obs is t_final and copy it instead.
    // state_int [systems, 2(slip/stress), 2(slip_components), patches]
    // sim_state [systems, t_steps, 2(slip/stress), 2(slip_components), patches]
    // only save velocity/slip rate (stress)
    // copy from updated systems, when systems_to_process< cuda_batch_size, restart copying from beginning
    copy_velocity(state_init, sim_state, num_inner_patches, num_t_obs,
        systems_to_process, cuda_batch_size, solver->threads);


    // cudaDeviceSynchronize();
    // for(auto i=0; i<cuda_batch_size; i++)
    //     std::cout << i<< " "
    //         << state_init[i*4*num_inner_patches] << " "
    //         << state_init[i*4*num_inner_patches + 2*num_inner_patches] << " "
    //         << sim_state[i*num_t_obs*4*num_inner_patches] << " "
    //         << sim_state[i*num_t_obs*4*num_inner_patches + 2*num_inner_patches] << "\n";

    // convert logarithmic velocity to linear one
    convert_slip_rate<T>(sim_state, num_forward_batch, num_t_obs, num_inner_patches, v_0, solver->threads);

    // call displacement routines - see details in Displacement.cuh for different implementations
    // assume Cd is a constant and gf is time independent
    compute_displacement_impl1<T>(
        obs_disp, sim_state, G_surf, num_forward_batch, num_t_obs, num_inner_patches, 3 * num_stations, v_0,
        (T) 1.0, (T) 0.0, solver->threads); // alpha beta for gemm C = alpha A B + beta C

    /*
    // displacement subtraction from t_num_eq
    std::cout << "subtract displacement " << num_forward_batch << " "
        << num_t_obs << " " << n_slips_obs << "\n";
    cudaDeviceSynchronize();
    for(auto i=0; i< n_slips_obs; i++)
        std::cout << "t_eq " << i  << ": " << i_slips_obs[i] << "\n";
    auto n_observations =  3 * num_stations;
    // debug for two observations before subtraction
    std::cout << obs_disp[i_slips_obs[0]*n_observations] << ": " << obs_disp[(i_slips_obs[0]+1)*n_observations] << "\n";
    std::cout << obs_disp[i_slips_obs[0]*n_observations+n_observations/2] << ": "
        << obs_disp[(i_slips_obs[0]+2)*n_observations+n_observations/2] << "\n";
    */

    if (n_slips_obs > 0)
        subtract_displacement_from_teq(
            obs_disp, num_forward_batch, num_t_obs, 3 * num_stations,
            i_slips_obs, n_slips_obs, solver->threads);

    /*
    // debug for two observations after subtraction
    cudaDeviceSynchronize();
    std::cout << obs_disp[i_slips_obs[0]*n_observations] << ": " << obs_disp[(i_slips_obs[0]+1)*n_observations] << "\n";
    std::cout << obs_disp[i_slips_obs[0]*n_observations+n_observations/2] << ": "
        << obs_disp[(i_slips_obs[0]+2)*n_observations+n_observations/2] << "\n";
    */

}

// size estimation methods

// estimate object size
template <typename T>
RateDependent<T>::size_type RateDependent<T>::estimate_object_size(const int num_ix_eq, const int n_slips_obs,
                                            const int num_t_obs, const int num_inner_patches, const int UNITS,
                                            const int cuda_batch_size, const int num_forward_batch,
                                            const int num_eq, const int num_stations) {
    size_type size_model = (2 * sizeof(bool) +
                       (15 +
                        num_ix_eq +
                        n_slips_obs +
                        n_slips_obs) * sizeof(int) +
                       (6 +
                        num_t_obs +
                        (num_ix_eq + 2) +
                        ((size_type)num_inner_patches * 2 * (size_type)num_inner_patches * 2) +
                        ((size_type)num_inner_patches * 2) +
                        ((size_type)num_inner_patches * 2) +
                        ((size_type)UNITS * (size_type)num_inner_patches) +
                        ((size_type)cuda_batch_size * (size_type)num_t_obs * (size_type)UNITS * (size_type)num_inner_patches)) * sizeof(T));
    size_type size_forward = (1 * sizeof(int) +
                         (((size_type)num_forward_batch * (size_type)num_inner_patches) +
                          ((size_type)num_forward_batch * (size_type)num_eq * (size_type)num_inner_patches * 2) +
                          (2 * (size_type)num_inner_patches * 3 * (size_type)num_stations) +
                          ((size_type)num_forward_batch * (size_type)num_t_obs * 3 * (size_type)num_stations) +
                          (5 * (size_type)num_inner_patches * (size_type)UNITS * (size_type)num_forward_batch) +
                          (10 * (size_type)num_inner_patches * (size_type)UNITS * (size_type)num_forward_batch) +
                          ((size_type)num_forward_batch * (size_type)num_t_obs * (size_type)num_inner_patches * 2)) * sizeof(T));
    return size_model + size_forward;
}

// explicit instantiation
template class altar::models::seas::cuda::ratedependent::RateDependent<float>;
template class altar::models::seas::cuda::ratedependent::RateDependent<double>;

} // end of namespace
// end of file
