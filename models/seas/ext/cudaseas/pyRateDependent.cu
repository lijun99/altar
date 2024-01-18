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
        int num_systems,
        int systems_batch,
        int max_cycles,
        int num_t_obs,
        py::capsule t_obs_sec,
        int num_ix_eq,
        int num_eq,
        py::capsule delta_tau_bounded_indices,
        // py::capsule ix_eq_joint,
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
        int num_stations
    )
    {
        // printf("inside pyRateDependent.cu:initialize\n");
        // initialize CUDA model, assuming all shapes are correct
        _cmodel->initialize(
            num_systems,
            systems_batch,
            max_cycles,
            num_t_obs,
            convertPyArray<T, cuda_vector>(t_obs_sec),
            num_ix_eq,
            num_eq,
            convertPyArray<int, cuda_vector>(delta_tau_bounded_indices),
            // convertPyArray<int, cuda_vector>(ix_eq_joint),
            convertPyArray<T, cuda_vector>(t_events),
            convertPyArray<int, cuda_vector>(i_slips_obs),
            n_slips_obs,
            v_0,
            mu_over_2vs,
            num_inner_patches,
            convertPyArray<T, cuda_vector>(K_inner_inner_onfault),
            convertPyArray<T, cuda_vector>(K_inner_asperities_v_plate),
            convertPyArray<T, cuda_vector>(v_plate_ddcs_proj_eff_inner),
            convertPyArray<T, cuda_vector>(state_init),
            convertPyArray<T, cuda_vector>(sim_state),
            atol,
            rtol,
            spinup_atol,
            spinup_rtol,
            num_stations
        );
    }

    // just pass forward model through
    void forward_model_batch(
        py::capsule alpha_h_vec,
        py::capsule delta_tau_div_alpha_h,
        py::capsule G_surf,
        py::capsule obs_disp,
        const int batches,
        const int num_threads = 0,
        const bool verbose = false)
    {
        // printf("inside pyRateDependent.cu:forward_model_batch\n");
        _cmodel->forward_model_batch(
            convertPyArray<T, cuda_vector>(alpha_h_vec), // (num_systems, num_inner_patches, ) [Pa]
            convertPyArray<T, cuda_vector>(delta_tau_div_alpha_h), // (num_systems, num_eq, num_inner_patches, 2) [-]
            convertPyArray<T, cuda_matrix>(G_surf), // (2*num_inner_patches, 3*num_stations) [-]
            convertPyArray<T, cuda_matrix>(obs_disp), // (num_systems, num_t_obs*3*num_stations) [m]
            batches, // batch size <=samples (in AlTar, not all samples are computed in simulations)
            num_threads, // number of threads 1 <= num_threads <= 5120, 0 means internally estimated
            verbose // whether to print info and progress indicators or not
        );
    }

    // estimator functions
    long estimate_model_size() {
        auto s = _cmodel->estimate_model_size();
        return s;
    }
    long estimate_forward_size(const int batches) {
        auto s = _cmodel->estimate_forward_size(batches);
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
             py::arg("G_surf"), py::arg("obs_disp"), py::arg("batches"),
             py::arg("num_threads") = 0, py::arg("verbose") = false)
        .def("estimate_model_size", &pyRateDependent_double::estimate_model_size)
        .def("estimate_forward_size", &pyRateDependent_double::estimate_forward_size);
    py::class_<pyRateDependent_float>(m, "model_float")
        .def(py::init())
        .def("initialize", &pyRateDependent_float::initialize)
        .def("forward_model_batch", &pyRateDependent_float::forward_model_batch,
             py::arg("alpha_h_vec"), py::arg("delta_tau_div_alpha_h"),
             py::arg("G_surf"), py::arg("obs_disp"), py::arg("batches"),
             py::arg("num_threads") = 0, py::arg("verbose") = false)
        .def("estimate_model_size", &pyRateDependent_float::estimate_model_size)
        .def("estimate_forward_size", &pyRateDependent_float::estimate_forward_size);
}

} // end of namespace pycuda::seas::ratedependent

// end of file
