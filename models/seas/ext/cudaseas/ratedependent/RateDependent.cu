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
        T* state_init_,
        T* sim_state_,
        T atol_,
        T rtol_,
        T spinup_atol_,
        T spinup_rtol_,
        int num_stations_,
        T* G_surf_,
        T* obs_disp_)
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
    G_surf = G_surf_;
    obs_disp = obs_disp_;
}

template <typename T>
void RateDependent<T>::set_system_odes(
    T* alpha_h_vec_,
    T* delta_tau_div_alpha_h_
    )
{
    // printf("inside RateDependent.cu:set_system_odes\n");
    // save system-specific rheology and event realization
    alpha_h_vec = alpha_h_vec_;
    delta_tau_div_alpha_h = delta_tau_div_alpha_h_;

    // printf("  assigned pointers\n");

    // create an instance of odefunc
    odefunc = new OdeType{num_inner_patches, UNITS, num_systems, alpha_h_vec, mu_over_2vs, v_0, K_inner_inner_onfault,
                          K_inner_asperities_v_plate, v_plate_ddcs_proj_eff_inner};

    // printf("  initialized odefunc\n");

    // create an instance of events (including starting/ending time)
    events = new EventType{num_ix_eq, num_eq, t_events, delta_tau_div_alpha_h, delta_tau_bounded_indices,
                           num_systems, num_inner_patches, UNITS};

    // printf("  initialized events\n");

    // create the solver
    solver = new SolverType{*odefunc, *events, atol, rtol, spinup_atol, spinup_rtol, systems_batch};

    // printf("  initialized solver\n");
    // t_eval_joint_sec [num_t_eval]
    // sim_state [num_t_eval, ]
    solver->set_dense_output(num_t_eval, t_eval_joint_sec, sim_state);

    // printf("  set dense output\n");
}

template <typename T>
void RateDependent<T>::forward_model_batch (
    T* predictions,  // [samples, t_steps, displacement_size]  displacement_size=stations*disp_components
    const T* theta,  // alpha_h_vec [samples, patches]
    const T* gf,     // [slip_size, displacement_size] slip_size = 2 * patches
    const int num_systems // batch size <=samples (in AlTar, not all samples are computed in simulations)
) {
    // printf("inside RateDependent.cu:forward_model_batch\n");

    // std::cout << "Debug forward_model_batch: "
    //     << "num_systems=" << num_systems
    //     << ", systems_batch=" << systems_batch
    //     << ", system_size=" << system_size
    //     << std::endl;

    // set theta as alpha_h in ode functions
    odefunc->alpha_h = theta;

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

    /*
    // convert exponential velocity to linear one
    for (auto isys = 0; isys < num_systems; isys++)
    for (auto iteval = 0; iteval < num_t_eval; iteval++)
    for (auto iunit = 2; iunit < 4; iunit++)
    for (auto ipatch = 0; ipatch < num_inner_patches; ipatch++)
    {
        // (num_systems, num_t_eval, UNITS * num_inner_patches)
        int i = isys * num_t_eval * UNITS * num_inner_patches
                + iteval * UNITS * num_inner_patches
                + iunit * num_inner_patches
                + ipatch;
        sim_state[i] = v_0 * exp(sim_state[i]);
    }
    */

    // call displacement routines - see details in Displacement.cuh for different implementations
    // assume Cd is a constant and gf is time independent
    compute_displacement_impl1(
        predictions, sim_state, gf, num_systems, num_t_eval, num_inner_patches, 3 * num_stations, v_0,
        (T)1.0, (T)0.0); // alpha beta for gemm C = alpha A B + beta C

}

// OBSOLETE
// need to rewrite to use block per system
template<typename T>
__global__
void compute_displacement_kernel(const T* sim_state, const T* G_surf, T* obs_disp,
        const int system_id, const int num_t_eval, const int num_inner_patches, const int num_stations,
        const int UNITS)
{
    // T* sim_state; // simulated state variables (num_systems, num_t_eval, UNITS * num_inner_patches) [m|m|-|-]
    // T* G_surf; // Displacement kernel for all stations (1, 2*num_inner_patches, 3*num_stations) [-]
    // T* obs_disp; // Surface observations for all stations (num_systems, num_t_eval, 3*num_stations) [m]

    // setup
    int i_time = blockIdx.y * blockDim.y + threadIdx.y;
    int i_station = blockIdx.x * blockDim.x + threadIdx.x;
    if ((i_time >= num_t_eval) || (i_station >= num_stations)) return;

    // calculate tensor product
    // loop over surface displacement components
    for (auto i_dispcomp = 0; i_dispcomp < 3; i_dispcomp++)
    {
        int obs_ind = system_id * num_t_eval * 3 * num_stations
                      + i_time * 3 * num_stations
                      + i_dispcomp * num_stations
                      + i_station;
        obs_disp[obs_ind] = 0.0;
        // loop over fault slip components
        for (auto i_slipcomp = 0; i_slipcomp < 2; i_slipcomp++)
        {
            // loop over fault patches
            for (auto i_patch = 0; i_patch < num_inner_patches; i_patch++)
            {
                obs_disp[obs_ind] += G_surf[i_slipcomp * num_inner_patches * 3 * num_stations
                                            + i_patch * 3 * num_stations
                                            + i_dispcomp * num_stations
                                            + i_station]
                                     * sim_state[system_id * num_t_eval * UNITS * num_inner_patches
                                                 + i_time * UNITS * num_inner_patches
                                                 + i_slipcomp * num_inner_patches
                                                 + i_patch];
            }
        }
    }
}

// CPU Routine - may be used as a check
// compute displacement from slips
// @note sim_state is arranged in shape (systems, times, 4*patches) - C-style
//       G_surf is arranged in shape (times, 2*patches, 3*stations)
//          - merged with data covariance, therefore, different for different t
//          TODO check if merging still makes sense
//       obs_disp is arranged in shape (systems, times, 3*stations)
template <typename T>
void RateDependent<T>::compute_displacement()
{
    // decide the execution size - one sample per thread
    // const int threadsPerBlock = 128;
    // const int numberOfBlocks = 8;
    for (auto system_id = 0; system_id < num_systems; system_id++)
    {
        // // call kernel
        // compute_displacement_kernel<T><<<numberOfBlocks, threadsPerBlock>>>(
        //     sim_state, G_surf, obs_disp,
        //     system_id, num_t_eval, num_inner_patches, num_stations, UNITS);
        // cudaDeviceSynchronize();
        
        // setup
        for (auto i_time = 0; i_time < num_t_eval; i_time++)
        {
            if (i_time % 100 == 0) printf("system=%i/%i, time=%i/%i\n", system_id + 1, num_systems, i_time + 1, num_t_eval);
            for (auto i_station = 0; i_station < num_stations; i_station++)
            {
                // calculate tensor product
                // loop over surface displacement components
                for (auto i_dispcomp = 0; i_dispcomp < 3; i_dispcomp++)
                {
                    int obs_ind = system_id * num_t_eval * 3 * num_stations
                                  + i_time * 3 * num_stations
                                  + i_dispcomp * num_stations
                                  + i_station;
                    obs_disp[obs_ind] = 0.0;
                    // printf("obs_disp[%i, %i, %i, %i]\n", system_id, i_time, i_dispcomp, i_station);
                    // loop over fault slip components
                    for (auto i_slipcomp = 0; i_slipcomp < 2; i_slipcomp++)
                    {
                        // loop over fault patches
                        for (auto i_patch = 0; i_patch < num_inner_patches; i_patch++)
                        {
                            // printf("G[%i, %i, %i, %i] = %g\n", i_slipcomp, i_patch, i_dispcomp, i_station,
                            //     G_surf[i_slipcomp * num_inner_patches * 3 * num_stations
                            //                             + i_patch * 3 * num_stations
                            //                             + i_dispcomp * num_stations
                            //                             + i_station]);
                            // printf("sim_state[%i, %i, %i, %i] = %g\n", system_id, i_time, i_slipcomp, i_patch,
                            //     sim_state[system_id * num_t_eval * UNITS * num_inner_patches
                            //                                 + i_time * UNITS * num_inner_patches
                            //                                 + i_slipcomp * num_inner_patches
                            //                                 + i_patch]);
                            obs_disp[obs_ind] += G_surf[i_slipcomp * num_inner_patches * 3 * num_stations
                                                        + i_patch * 3 * num_stations
                                                        + i_dispcomp * num_stations
                                                        + i_station]
                                                * sim_state[system_id * num_t_eval * UNITS * num_inner_patches
                                                            + i_time * UNITS * num_inner_patches
                                                            + i_slipcomp * num_inner_patches
                                                            + i_patch];
                        }
                    }
                }
            }
        }
    }
}

// explicit instantiation
template class altar::models::seas::cuda::ratedependent::RateDependent<float>;
template class altar::models::seas::cuda::ratedependent::RateDependent<double>;

} // end of namespace
// end of file
