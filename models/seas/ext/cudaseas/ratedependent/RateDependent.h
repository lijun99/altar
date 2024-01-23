// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2023 california institute of technology
// all rights reserved
//

// code guard
#if !defined(altar_models_seas_cuda_RateDependent_h)
#define altar_models_seas_cuda_RateDependent_h

// cuda ode solver
#include "cudaode.cuh"
// my definitions of ode and events (coseismic)
#include "Ode.cuh"
#include "Events.cuh"

// declaration
namespace altar::models::seas::cuda::ratedependent {

template<typename T>
class RateDependent {
    // methods
    public:

        // types
        using OdeType = RateDependentODE<T>; // ode function defition from Ode.cuh
        using EventType = SEASEvents<T>; // event(coseismic) from Events.cuh
        // this is an ode solver with spin up procedure built-in
        using SolverType = ::cuda::ode::dopri5::SpinupSolver<T, OdeType, EventType>;

        using size_type = std::size_t;

        RateDependent() = default; // default constructor
        ~RateDependent() = default; // default destructor

        // initial parameters and data
        void initialize(
            int cuda_batch_size_,
            int max_cycles_,
            int num_t_obs_,
            T* t_obs_sec_,
            int num_ix_eq_,
            int num_eq_,
            int* delta_tau_bounded_indices_,
            T* t_events_,
            int* i_slips_obs,
            int n_slips_obs,
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
            int num_stations_
        );

        // perform forward modeling
        void forward_model_batch(
            const T* alpha_h_vec, // (a-b)*sigma_E strength parameter on fault patches (num_forward_batch, num_inner_patches, ) [Pa]
            const T* delta_tau_div_alpha_h, // stress change for each system and earthquake divided by alpha_h (num_forward_batch, num_eq, num_inner_patches, 2) [-]
            const T* G_surf, // Displacement kernel for all stations (1, 2*num_inner_patches, 3*num_stations) [-]
            T* obs_disp,  // Surface observations for all stations (num_forward_batch, num_t_obs, 3*num_stations) [m]
            const int num_forward_batch, // batch size <=samples (in AlTar, not all samples are computed in simulations)
            const int num_threads, // number of threads 1 <= num_threads <= 1024, 0 means internally estimated
            bool verbose // whether to print info and progress indicators or not
        );

        // estimate object size
        static size_type estimate_object_size(const int num_ix_eq, const int n_slips_obs,
                                         const int num_t_obs, const int num_inner_patches, const int UNITS,
                                         const int cuda_batch_size, const int num_forward_batch,
                                         const int num_eq, const int num_stations);

    // parameters
    private:

        OdeType* odefunc;
        EventType* events;
        SolverType* solver;

        // general variables
        int cuda_batch_size; // maximum number of systems in a cuda batch[-]
        const bool DENSE_OUT = true; // always output the dense last cycle [-]

        // cycles
        int max_cycles; // maximum number of cycles to simulate for each system [-]
        int num_t_obs; // number of timesteps [-]
        T* t_obs_sec; // timesteps to simulate (num_t_obs, ) [s]

        // events
        int num_ix_eq; // = num_slips, number of non-unique earthquakes [-]
        int num_eq; // number of unique earthquakes [-]
        int* delta_tau_bounded_indices; // indices mapping the num_ix_eq event occurrences to the num_eq unique events (num_ix_eq, ) [-]
        T* t_events; // timestamps of start time, end time, and earthquakes (n_events = num_ix_eq + 2, ) [s]
        int* i_slips_obs; // indices of earthquakes in t_obs
        int n_slips_obs;  // number of observed indices

        // rheology
        const int UNITS = 4; // number of variables in each patch [-]
        const int components = 2 ; // along strike and dip directions
        T v_0; // logarithmic normalization velocity [m/s]
        T mu_over_2vs; // radiation damping term [Pa * s/m]

        // fault
        int num_inner_patches; // number of simulated patches [-]
        int system_size; // num_inner_patches * UNITS [-]
        T* K_inner_inner_onfault; // inner stress kernel (num_inner_patches, 2, num_inner_patches, 2) [Pa/m]
        T* K_inner_asperities_v_plate; // inner stressing rate from locked asperities (num_inner_patches, 2) [Pa/s]
        T* v_plate_ddcs_proj_eff_inner; // plate velocity on patches (num_inner_patches, 2) [m/s]
        T* state_init; // initial patch state,  cuda_batch_size*(4 * num_inner_patches) [m|m|-|-]

        // ode
        T atol; // absolute tolerance for ODE integrator [-]
        T rtol; // relative tolerance for ODE integrator [-]
        T spinup_atol; // absolute tolerance for spinup check [-]
        T spinup_rtol; // relative tolerance for spinup check [-]
        int conv_i_start; // starting index to check for spinup [-]
        int conv_i_stop; // stopping index to check for spinup (including) [-]
        T* sim_state; // simulated state variables (num_systems, num_t_obs, UNITS * num_inner_patches) [m|m|-|-]

        // observations
        int num_stations; // number of observers [-]

    }; //end of class RateDependent

} // end of namespace

#endif
// end of file
