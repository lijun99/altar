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

        RateDependent() = default; // default constructor
        ~RateDependent() = default; // default destructor

        // initial parameters and data
        void initialize(
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
            T spinup_rtol_
        );

        // set spin up data (initial values)
        void set_system_odes(
            T* alpha_h_vec_,
            T* delta_tau_bounded_compressed_
        );

        // perform forward modeling
        void forward_model_batch();

    // parameters
    private:

        OdeType* odefunc;
        EventType* events;
        SolverType* solver;

        // general variables
        int num_systems; // number of systems [-]
        int systems_batch; // number of num_systems to process in batch [-]
        const bool DENSE_OUT = true; // always output the dense last cycle [-]
        const bool USE_V_INIT_FOR_ALL = true; // always use a single v_init for all systems [-]

        // cycles
        int max_cycles; // maximum number of cycles to simulate for each system [-]
        int num_t_eval; // number of timesteps [-]
        T* t_eval_joint_sec; // timesteps to simulate (num_t_eval, ) [s]

        // events
        int num_ix_eq; // = num_slips, number of non-unique earthquakes [-]
        int num_eq; // number of unique earthquakes [-]
        T* delta_tau_div_alpha_h; // stress change for each system and earthquake divided by alpha_h (num_systems, num_eq, num_inner_patches, 2) [-]
        int* delta_tau_bounded_indices; // indices mapping the num_ix_eq event occurrences to the num_eq unique events (num_ix_eq, ) [-]
        int* ix_eq_joint; // indices of earthquakes in t_eval_joint_sec (num_ix_eq, ) [-]
        T* t_events; // timestamps of start time, end time, and earthquakes (n_events, ) [s]

        // rheology
        const int UNITS = 4; // number of variables in each patch [-]
        T v_0; // logarithmic normalization velocity [m/s]
        T mu_over_2vs; // radiation damping term [Pa * s/m]
        T* alpha_h_vec; // (a-b)*sigma_E strength parameter on fault patches (num_systems, num_inner_patches, ) [Pa]

        // fault
        int num_inner_patches; // number of simulated patches [-]
        int system_size; // num_inner_patches * UNITS [-]
        T* K_inner_inner_onfault; // inner stress kernel (num_inner_patches, 2, num_inner_patches, 2) [Pa/m]
        T* K_inner_asperities_v_plate; // inner stressing rate from locked asperities (num_inner_patches, 2) [Pa/s]
        T* v_plate_ddcs_proj_eff_inner; // plate velocity on patches (num_inner_patches, 2) [m/s]
        T* v_init; // initial patch velocities (num_inner_patches, 2) [m/s]

        // ode
        T atol; // absolute tolerance for ODE integrator [-]
        T rtol; // relative tolerance for ODE integrator [-]
        T spinup_atol; // absolute tolerance for spinup check [-]
        T spinup_rtol; // relative tolerance for spinup check [-]
        int conv_i_start; // starting index to check for spinup [-]
        int conv_i_stop; // stopping index to check for spinup (including) [-]
        T* sim_state; // simulated state variables (num_systems, num_t_eval, system_size) [m|m|-|-]

    }; //end of class RateDependent

} // end of namespace

#endif
// end of file
