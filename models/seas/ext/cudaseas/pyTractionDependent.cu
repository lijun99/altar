/**
 * pyTractionDependent.cu - python wrapper for Traction-Dependent ODE Solver
 */
#include <stdexcept>
#include <string>
#include <math.h>

// namespace setup
#include "external.h"
#include "forward.h"

// my definitions
#include "tractiondependent/TractionDependent.h"


namespace altar::cuda::py::seas::tractiondependent {

using size_type = std::size_t;

template <typename T, typename D>
T* convertPyArray(py::capsule pycap)
{
    auto cap = static_cast<D *>(pycap.get_pointer());
    return (T *)cap->data;
}


template<typename T>
class pyTractionDependent
{
    using model_type = altar::models::seas::cuda::tractiondependent::TractionDependent<T>;

private:
    model_type * _cmodel;

public:

    pyTractionDependent () {_cmodel = new model_type();}

    // initialize — note the additional tau_0 parameter
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
        T tau_0,              // <--- NEW: constant traction parameter
        int num_inner_patches,
        py::capsule K_inner_inner_onfault,
        py::capsule K_inner_asperities_v_plate,
        py::capsule v_plate_ddcs_proj_eff_inner,
        py::capsule sim_state,
        T atol,
        T rtol,
        T spinup_atol,
        T spinup_rtol,
        int num_stations,
        py::capsule obs_mask,
        py::capsule i_stat_ref,
        int n_stat_ref,
        int ref_vel_index
    )
    {
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
            tau_0,            // <--- NEW
            num_inner_patches,
            convertPyArray<T, cuda_vector>(K_inner_inner_onfault),
            convertPyArray<T, cuda_vector>(K_inner_asperities_v_plate),
            convertPyArray<T, cuda_vector>(v_plate_ddcs_proj_eff_inner),
            convertPyArray<T, cuda_vector>(sim_state),
            atol,
            rtol,
            spinup_atol,
            spinup_rtol,
            num_stations,
            convertPyArray<bool, cuda_vector>(obs_mask),
            convertPyArray<int, cuda_vector>(i_stat_ref),
            n_stat_ref,
            ref_vel_index
        );
    }

    void forward_model_batch(
        py::capsule state_init,
        py::capsule alpha_h_vec,
        py::capsule delta_tau_div_alpha_h,
        py::capsule delta_tau_bounded_indices,
        py::capsule delta_tau_bounded_indices_final,
        py::capsule G_surf,
        py::capsule obs_disp,
        py::capsule ref_obs,
        py::capsule obs_farfield,
        py::capsule obs_ep,
        const int batches,
        const T v_ratio_max,
        const int num_threads = 0,
        const bool verbose = false)
    {
        _cmodel->forward_model_batch(
            convertPyArray<T, cuda_matrix>(state_init),
            convertPyArray<T, cuda_vector>(alpha_h_vec),
            convertPyArray<T, cuda_vector>(delta_tau_div_alpha_h),
            convertPyArray<int, cuda_vector>(delta_tau_bounded_indices),
            convertPyArray<int, cuda_vector>(delta_tau_bounded_indices_final),
            convertPyArray<T, cuda_matrix>(G_surf),
            convertPyArray<T, cuda_matrix>(obs_disp),
            convertPyArray<T, cuda_matrix>(ref_obs),
            convertPyArray<T, cuda_vector>(obs_farfield),
            convertPyArray<T, cuda_vector>(obs_ep),
            batches,
            v_ratio_max,
            num_threads,
            verbose
        );
    }

    size_type estimate_object_size(const int num_ix_eq, const int n_slips_obs,
                              const int num_t_obs, const int num_inner_patches, const int UNITS,
                              const int cuda_batch_size, const int num_forward_batch,
                              const int num_eq, const int num_stations, const int n_stat_ref) {
        return model_type::estimate_object_size(num_ix_eq, n_slips_obs,
                                               num_t_obs, num_inner_patches, UNITS,
                                               cuda_batch_size, num_forward_batch,
                                               num_eq, num_stations, n_stat_ref);
    }
};

void
module(py::module & m)
{
    using pyTractionDependent_float = pyTractionDependent<float>;
    using pyTractionDependent_double = pyTractionDependent<double>;

    py::class_<pyTractionDependent_double>(m, "model_double")
        .def(py::init())
        .def("initialize", &pyTractionDependent_double::initialize)
        .def("forward_model_batch", &pyTractionDependent_double::forward_model_batch,
             py::arg("state_init"), py::arg("alpha_h_vec"), py::arg("delta_tau_div_alpha_h"),
             py::arg("delta_tau_bounded_indices"), py::arg("delta_tau_bounded_indices_final"),
             py::arg("G_surf"), py::arg("obs_disp"), py::arg("ref_obs"), py::arg("obs_farfield"), py::arg("obs_ep"),
             py::arg("batches"), py::arg("v_ratio_max") = 0, py::arg("num_threads") = 0, py::arg("verbose") = false)
        .def("estimate_object_size", &pyTractionDependent_double::estimate_object_size);
    py::class_<pyTractionDependent_float>(m, "model_float")
        .def(py::init())
        .def("initialize", &pyTractionDependent_float::initialize)
        .def("forward_model_batch", &pyTractionDependent_float::forward_model_batch,
             py::arg("state_init"), py::arg("alpha_h_vec"), py::arg("delta_tau_div_alpha_h"),
             py::arg("delta_tau_bounded_indices"), py::arg("delta_tau_bounded_indices_final"),
             py::arg("G_surf"), py::arg("obs_disp"), py::arg("ref_obs"), py::arg("obs_farfield"), py::arg("obs_ep"),
             py::arg("batches"), py::arg("v_ratio_max") = 0, py::arg("num_threads") = 0, py::arg("verbose") = false)
        .def("estimate_object_size", &pyTractionDependent_float::estimate_object_size);
}

} // end of namespace
