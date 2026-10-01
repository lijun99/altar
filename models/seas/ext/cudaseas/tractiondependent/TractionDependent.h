// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2023 california institute of technology
// all rights reserved
//

// code guard
#if !defined(altar_models_seas_cuda_TractionDependent_h)
#define altar_models_seas_cuda_TractionDependent_h

// cuda ode solver
#include "cudaode.cuh"
// my definitions of ode and events
#include "Ode.cuh"
#include "Events.cuh"

// declaration
namespace altar::models::seas::cuda::tractiondependent {

template<typename T, class MethodType = ::cuda::ode::dopri5::Dopri5<T>>
class TractionDependent {
    public:

        // types
        using OdeType = TractionDependentODE<T>;
        using EventType = SEASEvents<T>;
        using SolverType = ::cuda::ode::dopri5::SpinupSolver<T, OdeType, EventType, MethodType>;

        using size_type = std::size_t;

        TractionDependent() = default;
        ~TractionDependent() = default;

        // initial parameters and data
        void initialize(
            int cuda_batch_size_,
            int max_cycles_,
            int num_t_obs_,
            T* t_obs_sec_,
            int num_ix_eq_,
            int num_eq_,
            T* t_events_,
            int* i_slips_obs,
            int n_slips_obs,
            T v_0_,
            T mu_over_2vs_,
            T tau_0_,          // <--- NEW: constant traction parameter
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
            int ref_vel_index_
        );

        // perform forward modeling
        void forward_model_batch(
            T* state_init_,
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
            bool verbose
        );

        // estimate object size
        static size_type estimate_object_size(const int num_ix_eq, const int n_slips_obs,
                                              const int num_t_obs, const int num_inner_patches, const int UNITS,
                                              const int cuda_batch_size, const int num_forward_batch,
                                              const int num_eq, const int num_stations, const int n_stat_ref);

        // the step statistics of each system in the last batch
        std::vector<::cuda::ode::dopri5::StepStatistics> statistics;

    private:

        OdeType* odefunc;
        EventType* events;
        SolverType* solver;

        // general variables
        int cuda_batch_size;
        const bool DENSE_OUT = true;

        // cycles
        int max_cycles;
        int num_t_obs;
        T* t_obs_sec;

        // events
        int num_ix_eq;
        int num_eq;
        T* t_events;
        int* i_slips_obs;
        int n_slips_obs;

        // rheology
        const int UNITS = 4;
        const int components = 2;
        T v_0;
        T mu_over_2vs;
        T tau_0; // <--- NEW: constant traction parameter [Pa]

        // fault
        int num_inner_patches;
        int system_size;
        T* K_inner_inner_onfault;
        T* K_inner_asperities_v_plate;
        T* v_plate_ddcs_proj_eff_inner;

        // ode
        T atol;
        T rtol;
        T spinup_atol;
        T spinup_rtol;
        int conv_i_start;
        int conv_i_stop;
        T* sim_state;

        // observations
        int num_stations;
        bool* obs_mask;
        int* i_stat_ref;
        int n_stat_ref;
        int ref_vel_index;

    }; // end of class TractionDependent

} // end of namespace

#endif
// end of file
