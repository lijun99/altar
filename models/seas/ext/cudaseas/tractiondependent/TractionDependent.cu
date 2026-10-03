// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2023 california institute of technology
// all rights reserved
//

// for the build system
#include <portinfo>

// get my class declaration
#include "TractionDependent.h"

// get displacement routines
#include "Displacement.cuh"

#include <iostream>
#include <cassert>

namespace altar::models::seas::cuda::tractiondependent {

// Initialize model parameters
template <typename T, class MethodType>
void TractionDependent<T, MethodType>::initialize(
        int cuda_batch_size_,
        int max_cycles_,
        int num_t_obs_,
        T* t_obs_sec_,
        int num_ix_eq_,
        int num_eq_,
        T* t_events_,
        int* i_slips_obs_,
        int n_slips_obs_,
        T v_0_,
        T mu_over_2vs_,
        T rho_,
        int num_inner_patches_,
        T* K_inner_inner_onfault_,
        T* K_inner_asperities_v_plate_,
        T* v_plate_ddcs_proj_eff_inner_,
        T* sim_state_,
        T atol_,
        T rtol_,
        T spinup_atol_,
        T spinup_rtol_,
        int num_stations_,
        bool* obs_mask_,
        int* i_stat_ref_,
        int n_stat_ref_,
        int ref_vel_index_)
{
    cuda_batch_size  = cuda_batch_size_;

    max_cycles = max_cycles_;
    num_t_obs = num_t_obs_;
    t_obs_sec = t_obs_sec_;

    num_ix_eq = num_ix_eq_;
    num_eq = num_eq_;
    t_events = t_events_;
    i_slips_obs = i_slips_obs_;
    n_slips_obs = n_slips_obs_;

    v_0 = v_0_;
    mu_over_2vs = mu_over_2vs_;
    rho = rho_;

    num_inner_patches = num_inner_patches_;
    system_size = num_inner_patches * UNITS;
    K_inner_inner_onfault = K_inner_inner_onfault_;
    K_inner_asperities_v_plate = K_inner_asperities_v_plate_;
    v_plate_ddcs_proj_eff_inner = v_plate_ddcs_proj_eff_inner_;
    // state_init = state_init_; // <--- needs to be in the forward model now

    atol = atol_;
    rtol = rtol_;
    spinup_atol = spinup_atol_;
    spinup_rtol = spinup_rtol_;
    conv_i_start = (UNITS / 2) * num_inner_patches;
    conv_i_stop = UNITS * num_inner_patches - 1;
    sim_state = sim_state_;

    num_stations = num_stations_;
    obs_mask = obs_mask_;
    i_stat_ref = i_stat_ref_;
    n_stat_ref = n_stat_ref_;
    ref_vel_index = ref_vel_index_;
}

template <typename T, class MethodType>
void TractionDependent<T, MethodType>::forward_model_batch(
    T* state_init,
    const T* alpha_h_vec,
    const T* delta_tau_div_alpha_h,
    int* delta_tau_bounded_indices,
    int* delta_tau_bounded_indices_final,
    const T* G_surf,
    T* obs_disp,
    T* ref_obs,
    const T* obs_farfield,
    const T* obs_ep,
    const int num_forward_batch,
    const T v_ratio_max,
    const int num_threads,
    bool verbose)
{

    assert(num_forward_batch <= cuda_batch_size);

    // create an instance of odefunc
    odefunc = new OdeType{num_inner_patches, UNITS, num_forward_batch,
                          alpha_h_vec, mu_over_2vs, v_0, rho,
                          K_inner_inner_onfault,
                          K_inner_asperities_v_plate,
                          v_plate_ddcs_proj_eff_inner};

    // create an instance of events
    events = new EventType{num_ix_eq, num_eq, t_events,
                           delta_tau_div_alpha_h,
                           delta_tau_bounded_indices,
                           delta_tau_bounded_indices_final,
                           num_forward_batch, num_inner_patches, UNITS,
                           v_ratio_max, alpha_h_vec, mu_over_2vs, v_0, rho};

    // create the solver
    solver = new SolverType{*odefunc, *events, atol, rtol,
                            spinup_atol, spinup_rtol,
                            num_forward_batch, num_threads};
    solver->set_dense_output(num_t_obs, t_obs_sec, sim_state);

    int system_offset = 0;
    auto systems_to_process = num_forward_batch;

    solver->set_init_values(state_init, false, systems_to_process, system_offset);
    solver->solve_ivp_cycles(DENSE_OUT, systems_to_process, system_offset,
                             conv_i_start, conv_i_stop, max_cycles, verbose);
    cudaDeviceSynchronize();

    // copy traction for next batch
    copy_traction(state_init, sim_state, num_inner_patches, num_t_obs,
        systems_to_process, cuda_batch_size, solver->threads);

    // convert traction to velocity
    convert_traction<T>(sim_state, num_forward_batch, num_t_obs,
                         num_inner_patches, v_0, mu_over_2vs, rho, alpha_h_vec, solver->threads);

    // compute displacement
    compute_displacement_impl1<T>(
        obs_disp, sim_state, G_surf, num_forward_batch, num_t_obs,
        num_inner_patches, 3 * num_stations, v_0,
        (T) 1.0, (T) 0.0, solver->threads);

    if (ref_vel_index >= 0) {
        remove_reference_surface_velocities<T>(
            obs_disp, sim_state, G_surf, t_obs_sec, num_forward_batch, num_t_obs,
            num_inner_patches, num_stations, ref_vel_index, solver->threads);
    }
    else {
        add_farfield_effects(obs_disp, obs_farfield, num_forward_batch,
                             num_t_obs, num_stations, solver->threads);
    }
    cudaDeviceSynchronize();

    // add euler pole motion
    add_euler_pole_motion(obs_disp, obs_ep, num_forward_batch,
                          num_t_obs, num_stations, solver->threads);
    cudaDeviceSynchronize();

    // reference / subtract / reset
    reference_subtract_reset_displacements(
        obs_disp, ref_obs, num_forward_batch, num_t_obs, num_stations,
        i_slips_obs, n_slips_obs, obs_mask, i_stat_ref, n_stat_ref,
        solver->threads);

    // keep the step statistics, before the solver goes away
    statistics = solver->statistics(num_forward_batch);

    // clean up
    if(solver != nullptr) { delete solver; solver = nullptr; }
    events->deallocate();
    if(events != nullptr) { delete events; events = nullptr; }
    if(odefunc != nullptr) { delete odefunc; odefunc = nullptr; }
}

// size estimation (same as ratedependent)
template <typename T, class MethodType>
TractionDependent<T, MethodType>::size_type TractionDependent<T, MethodType>::estimate_object_size(
    const int num_ix_eq, const int n_slips_obs,
    const int num_t_obs, const int num_inner_patches, const int UNITS,
    const int cuda_batch_size, const int num_forward_batch,
    const int num_eq, const int num_stations, const int n_stat_ref) {
    size_type size_model = (
        (2 + ((size_type)num_t_obs * 3 * (size_type)num_stations)) * sizeof(bool) +
        (16 + n_slips_obs + n_slips_obs + n_stat_ref + 1) * sizeof(int) +
        (6 + num_t_obs + (num_ix_eq + 2) +
         ((size_type)num_inner_patches * 2 * (size_type)num_inner_patches * 2) +
         ((size_type)num_inner_patches * 2) +
         ((size_type)num_inner_patches * 2) +
         ((size_type)UNITS * (size_type)num_inner_patches) +
         ((size_type)cuda_batch_size * (size_type)num_t_obs * (size_type)UNITS * (size_type)num_inner_patches)) * sizeof(T));
    size_type size_forward = (
        num_forward_batch * sizeof(bool) +
        (1 + 2 * num_ix_eq) * sizeof(int) +
        (((size_type)num_forward_batch * (size_type)num_inner_patches) +
         ((size_type)num_forward_batch * (size_type)num_eq * (size_type)num_inner_patches * 2) +
         (2 * (size_type)num_inner_patches * 3 * (size_type)num_stations) +
         ((size_type)(num_forward_batch + 1) * (size_type)num_t_obs * 3 * (size_type)num_stations) +
         (5 * (size_type)num_inner_patches * (size_type)UNITS * (size_type)num_forward_batch) +
         (10 * (size_type)num_inner_patches * (size_type)UNITS * (size_type)num_forward_batch) +
         ((size_type)num_forward_batch * (size_type)num_t_obs * (size_type)num_inner_patches * 2) +
         ((size_type)num_forward_batch * (size_type)num_t_obs * 3) +
         (size_type)num_forward_batch * 3 * (size_type)num_stations + 1) * sizeof(T));
    return size_model + size_forward;
}

// explicit instantiation
template class altar::models::seas::cuda::tractiondependent::TractionDependent<float>;
template class altar::models::seas::cuda::tractiondependent::TractionDependent<float, ::cuda::ode::radau5::Radau5<float>>;
template class altar::models::seas::cuda::tractiondependent::TractionDependent<double>;
template class altar::models::seas::cuda::tractiondependent::TractionDependent<double, ::cuda::ode::radau5::Radau5<double>>;

} // end of namespace
