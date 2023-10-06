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
        int num_t_eval,
        py::capsule t_eval_joint_sec,
        int num_ix_eq,
        int num_eq,
        py::capsule delta_tau_bounded_indices,
        py::capsule ix_eq_joint,
        py::capsule t_events,
        T v_0,
        T mu_over_2vs,
        int num_inner_patches,
        py::capsule K_inner_inner_onfault,
        py::capsule K_inner_asperities_v_plate,
        py::capsule v_plate_ddcs_proj_eff_inner,
        py::capsule v_init,
        py::capsule sim_state,
        T atol,
        T rtol,
        T spinup_atol,
        T spinup_rtol
    )
    {
        printf("inside pyRateDependent.cu:initialize\n");
        // initialize CUDA model, assuming all shapes are correct
        _cmodel->initialize(
            num_systems,
            systems_batch,
            max_cycles,
            num_t_eval,
            convertPyArray<T, cuda_vector>(t_eval_joint_sec),
            num_ix_eq,
            num_eq,
            convertPyArray<int, cuda_vector>(delta_tau_bounded_indices),
            convertPyArray<int, cuda_vector>(ix_eq_joint),
            convertPyArray<T, cuda_vector>(t_events),
            v_0,
            mu_over_2vs,
            num_inner_patches,
            convertPyArray<T, cuda_vector>(K_inner_inner_onfault),
            convertPyArray<T, cuda_vector>(K_inner_asperities_v_plate),
            convertPyArray<T, cuda_vector>(v_plate_ddcs_proj_eff_inner),
            convertPyArray<T, cuda_vector>(v_init),
            convertPyArray<T, cuda_vector>(sim_state),
            atol,
            rtol,
            spinup_atol,
            spinup_rtol
        );
    }

    // set system ODEs
    void set_system_odes(py::capsule alpha_h_vec, py::capsule delta_tau_div_alpha_h)
    {
        printf("inside pyRateDependent.cu:set_system_odes\n");
        // set internal values, assume shapes are matching
        _cmodel->set_system_odes(
            convertPyArray<T, cuda_vector>(alpha_h_vec),
            convertPyArray<T, cuda_vector>(delta_tau_div_alpha_h)
        );
    }

    // just pass forward model through
    void forward_model_batch()
    {
        printf("inside pyRateDependent.cu:forward_model_batch\n");
        _cmodel->forward_model_batch();
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
        .def("set_system_odes", &pyRateDependent_double::set_system_odes)
        .def("forward_model_batch", &pyRateDependent_double::forward_model_batch);
    py::class_<pyRateDependent_float>(m, "model_float")
        .def(py::init())
        .def("initialize", &pyRateDependent_float::initialize)
        .def("set_system_odes", &pyRateDependent_float::set_system_odes)
        .def("forward_model_batch", &pyRateDependent_float::forward_model_batch);
}

} // end of namespace pycuda::seas::ratedependent

// end of file
