/**
 * pyRateDependent.cc - python wrapper for Rate-Dependent ODE Solver
 **/
#include <stdexcept>
#include <string>
#include <math.h>

// namespace setup
#include "external.h"
#include "forward.h"

// my definitions
#include "ratedependent/RateDependent.h"
#include "statistics.h"


namespace altar::cuda::py::seas::ratedependent {

using size_type = std::size_t;

template<typename T, class MethodType = ::cuda::ode::dopri5::Dopri5<T>>
class pyRateDependent
{
    using model_type = altar::models::seas::cuda::ratedependent::RateDependent<T, MethodType>;

private:

    // model instance
    model_type * _cmodel;

public:

    // constructor
    pyRateDependent () {_cmodel = new model_type();}

    // the step statistics of the last batch
    py::dict step_statistics() { return statistics_dict(_cmodel->statistics); }

    // initialize cmodel parameters
    void initialize(
        int cuda_batch_size,
        int max_cycles,
        int num_t_obs,
        grid_t & t_obs_sec,
        int num_ix_eq,
        int num_eq,
        grid_t & t_events,
        grid_t & i_slips_obs,
        int n_slips_obs,
        T v_0,
        T mu_over_2vs,
        int num_inner_patches,
        grid_t & K_inner_inner_onfault,
        grid_t & K_inner_asperities_v_plate,
        grid_t & v_plate_ddcs_proj_eff_inner,
        grid_t & state_init,
        grid_t & sim_state,
        T atol,
        T rtol,
        T spinup_atol,
        T spinup_rtol,
        int num_stations,
        grid_t & obs_mask,
        grid_t & i_stat_ref,
        int n_stat_ref,
        int ref_vel_index
    )
    {
        // printf("inside pyRateDependent.cu:initialize\n");
        // initialize CUDA model, assuming all shapes are correct
        _cmodel->initialize(
            cuda_batch_size,
            max_cycles,
            num_t_obs,
            cells<T>(t_obs_sec),
            num_ix_eq,
            num_eq,
            cells<T>(t_events),
            cells<int>(i_slips_obs),
            n_slips_obs,
            v_0,
            mu_over_2vs,
            num_inner_patches,
            cells<T>(K_inner_inner_onfault),
            cells<T>(K_inner_asperities_v_plate),
            cells<T>(v_plate_ddcs_proj_eff_inner),
            cells<T>(state_init),
            cells<T>(sim_state),
            atol,
            rtol,
            spinup_atol,
            spinup_rtol,
            num_stations,
            cells<bool>(obs_mask),
            cells<int>(i_stat_ref),
            n_stat_ref,
            ref_vel_index
        );
    }

    // just pass forward model through
    void forward_model_batch(
        grid_t & alpha_h_vec,
        grid_t & delta_tau_div_alpha_h,
        grid_t & delta_tau_bounded_indices,
        grid_t & delta_tau_bounded_indices_final,
        grid_t & G_surf,
        grid_t & obs_disp,
        grid_t & ref_obs,
        grid_t & obs_farfield,
        grid_t & obs_ep,
        const int batches,
        const T v_ratio_max,
        const int num_threads = 0,
        const bool verbose = false)
    {
        // printf("inside pyRateDependent.cu:forward_model_batch\n");
        _cmodel->forward_model_batch(
            cells<T>(alpha_h_vec), // (num_systems, num_inner_patches, ) [Pa]
            cells<T>(delta_tau_div_alpha_h), // (num_systems, num_eq, num_inner_patches, 2) [-]
            cells<int>(delta_tau_bounded_indices), // indices mapping the num_ix_eq event occurrences to the num_eq unique events (num_ix_eq, ) [-]
            cells<int>(delta_tau_bounded_indices_final), // same as before but for the last, spun-up cycle
            cells<T>(G_surf), // (2*num_inner_patches, 3*num_stations) [-]
            cells<T>(obs_disp), // (num_systems, num_t_obs*3*num_stations) [m]
            cells<T>(ref_obs), // (num_systems, num_t_obs*3) [m]
            cells<T>(obs_farfield), // (num_t_obs*3*num_stations) [m]
            cells<T>(obs_ep), // (num_forward_batch, num_t_obs, 2, num_stations) [m]
            batches, // batch size <=samples (in AlTar, not all samples are computed in simulations)
            v_ratio_max, // ratio between maximum allowed velocity and reference velocity [-], zero if no maximum
            num_threads, // number of threads 1 <= num_threads <= 5120, 0 means internally estimated
            verbose // whether to print info and progress indicators or not
        );
    }

    // estimator functions
    size_type estimate_object_size(const int num_ix_eq, const int n_slips_obs,
                              const int num_t_obs, const int num_inner_patches, const int UNITS,
                              const int cuda_batch_size, const int num_forward_batch,
                              const int num_eq, const int num_stations, const int n_stat_ref) {
        auto s = _cmodel->estimate_object_size(num_ix_eq, n_slips_obs,
                                               num_t_obs, num_inner_patches, UNITS,
                                               cuda_batch_size, num_forward_batch,
                                               num_eq, num_stations, n_stat_ref);
        return s;
    }
};

// bind one instantiation of the model as {name}
template <class P>
void
bind(py::module & m, const char * name)
{
    py::class_<P>(m, name)
        .def(py::init())
        .def("step_statistics", &P::step_statistics)
        .def("initialize", &P::initialize,
             py::arg("cuda_batch_size"), py::arg("max_cycles"), py::arg("num_t_obs"), py::arg("t_obs_sec"),
             py::arg("num_ix_eq"), py::arg("num_eq"), py::arg("t_events"), py::arg("i_slips_obs"),
             py::arg("n_slips_obs"), py::arg("v_0"), py::arg("mu_over_2vs"),
             py::arg("num_inner_patches"), py::arg("K_inner_inner_onfault"),
             py::arg("K_inner_asperities_v_plate"), py::arg("v_plate_ddcs_proj_eff_inner"), py::arg("state_init"),
             py::arg("sim_state"), py::arg("atol"), py::arg("rtol"), py::arg("spinup_atol"),
             py::arg("spinup_rtol"), py::arg("num_stations"), py::arg("obs_mask"),
             py::arg("i_stat_ref"), py::arg("n_stat_ref"), py::arg("ref_vel_index"))
        .def("forward_model_batch", &P::forward_model_batch,
             py::arg("alpha_h_vec"), py::arg("delta_tau_div_alpha_h"),
             py::arg("delta_tau_bounded_indices"), py::arg("delta_tau_bounded_indices_final"),
             py::arg("G_surf"), py::arg("obs_disp"), py::arg("ref_obs"), py::arg("obs_farfield"), py::arg("obs_ep"),
             py::arg("batches"), py::arg("v_ratio_max") = 0, py::arg("num_threads") = 0, py::arg("verbose") = false)
        .def("estimate_object_size", &P::estimate_object_size);
}

// add bindings for the various cuda struct
void
module(py::module & m)
{
    bind<pyRateDependent<double>>(m, "model_double");
    bind<pyRateDependent<float>>(m, "model_float");
}

} // end of namespace pycuda::seas::ratedependent

// end of file
