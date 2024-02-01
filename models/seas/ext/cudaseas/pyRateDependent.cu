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


namespace altar::cuda::py::seas::ratedependent {

using size_type = std::size_t;

template <typename T, typename D>
T* convertPyArray(py::capsule pycap)
{
    auto cap = static_cast<D *>(pycap.get_pointer());
    return (T *)cap->data;
}


template<typename T>
class pyRateDependent
{
    using model_type = altar::models::seas::cuda::ratedependent::RateDependent<T>;

private:

    // model instance
    model_type * _cmodel;

public:

    // constructor
    pyRateDependent () {_cmodel = new model_type();}

    // initialize cmodel parameters
    void initialize(
        int cuda_batch_size,
        int max_cycles,
        int num_t_obs,
        py::capsule t_obs_sec,
        int num_ix_eq,
        int num_eq,
        py::capsule t_events,
        py::capsule i_slips_obs,
        int n_slips_obs,
        T v_0,
        T mu_over_2vs,
        int num_inner_patches,
        py::capsule K_inner_inner_onfault,
        py::capsule K_inner_asperities_v_plate,
        py::capsule v_plate_ddcs_proj_eff_inner,
        py::capsule state_init,
        py::capsule sim_state,
        T atol,
        T rtol,
        T spinup_atol,
        T spinup_rtol,
        int num_stations,
        py::capsule obs_mask,
        py::capsule i_stat_ref,
        int n_stat_ref
    )
    {
        // printf("inside pyRateDependent.cu:initialize\n");
        // initialize CUDA model, assuming all shapes are correct
        _cmodel->initialize(
            cuda_batch_size,
            max_cycles,
            num_t_obs,
            convertPyArray<T, cuda_vector>(t_obs_sec),
            num_ix_eq,
            num_eq,
            convertPyArray<T, cuda_vector>(t_events),
            convertPyArray<int, cuda_vector>(i_slips_obs),
            n_slips_obs,
            v_0,
            mu_over_2vs,
            num_inner_patches,
            convertPyArray<T, cuda_vector>(K_inner_inner_onfault),
            convertPyArray<T, cuda_vector>(K_inner_asperities_v_plate),
            convertPyArray<T, cuda_vector>(v_plate_ddcs_proj_eff_inner),
            convertPyArray<T, cuda_matrix>(state_init),
            convertPyArray<T, cuda_vector>(sim_state),
            atol,
            rtol,
            spinup_atol,
            spinup_rtol,
            num_stations,
            convertPyArray<bool, cuda_vector>(obs_mask),
            convertPyArray<int, cuda_vector>(i_stat_ref),
            n_stat_ref
        );
    }

    // just pass forward model through
    void forward_model_batch(
        py::capsule alpha_h_vec,
        py::capsule delta_tau_div_alpha_h,
        py::capsule delta_tau_bounded_indices,
        py::capsule delta_tau_bounded_indices_final,
        py::capsule G_surf,
        py::capsule obs_disp,
        py::capsule obs_farfield,
        const int batches,
        const int num_threads = 0,
        const bool verbose = false)
    {
        // printf("inside pyRateDependent.cu:forward_model_batch\n");
        _cmodel->forward_model_batch(
            convertPyArray<T, cuda_vector>(alpha_h_vec), // (num_systems, num_inner_patches, ) [Pa]
            convertPyArray<T, cuda_vector>(delta_tau_div_alpha_h), // (num_systems, num_eq, num_inner_patches, 2) [-]
            convertPyArray<int, cuda_vector>(delta_tau_bounded_indices), // indices mapping the num_ix_eq event occurrences to the num_eq unique events (num_ix_eq, ) [-]
            convertPyArray<int, cuda_vector>(delta_tau_bounded_indices_final), // same as before but for the last, spun-up cycle
            convertPyArray<T, cuda_matrix>(G_surf), // (2*num_inner_patches, 3*num_stations) [-]
            convertPyArray<T, cuda_matrix>(obs_disp), // (num_systems, num_t_obs*3*num_stations) [m]
            convertPyArray<T, cuda_vector>(obs_farfield), // (num_t_obs*3*num_stations) [m]
            batches, // batch size <=samples (in AlTar, not all samples are computed in simulations)
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

// add bindings for the various cuda struct
void
module(py::module & m)
{
    using pyRateDependent_float = pyRateDependent<float>;
    using pyRateDependent_double = pyRateDependent<double>;

    py::class_<pyRateDependent_double>(m, "model_double")
        .def(py::init())
        .def("initialize", &pyRateDependent_double::initialize)
        .def("forward_model_batch", &pyRateDependent_double::forward_model_batch,
             py::arg("alpha_h_vec"), py::arg("delta_tau_div_alpha_h"),
             py::arg("delta_tau_bounded_indices"), py::arg("delta_tau_bounded_indices_final"),
             py::arg("G_surf"), py::arg("obs_disp"), py::arg("obs_farfield"),
             py::arg("batches"), py::arg("num_threads") = 0, py::arg("verbose") = false)
        .def("estimate_object_size", &pyRateDependent_double::estimate_object_size);
    py::class_<pyRateDependent_float>(m, "model_float")
        .def(py::init())
        .def("initialize", &pyRateDependent_float::initialize)
        .def("forward_model_batch", &pyRateDependent_float::forward_model_batch,
             py::arg("alpha_h_vec"), py::arg("delta_tau_div_alpha_h"),
             py::arg("delta_tau_bounded_indices"), py::arg("delta_tau_bounded_indices_final"),
             py::arg("G_surf"), py::arg("obs_disp"), py::arg("obs_farfield"),
             py::arg("batches"), py::arg("num_threads") = 0, py::arg("verbose") = false)
        .def("estimate_object_size", &pyRateDependent_float::estimate_object_size);
}

} // end of namespace pycuda::seas::ratedependent

// end of file
