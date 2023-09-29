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
    // some shape parameters for pre-check
    int _num_t_eval;
    int _num_ix_eq;
    int _num_inner_patches;
    int _num_eq;
public:
    // constructor
    pyRateDependent () {_cmodel = new model_type();}
    // initialize cmodel parameters
    void initialize(
        int num_systems,
        int systems_batch,
        int max_cycles,
        py::capsule t_eval_joint_sec,
        int num_eq,
        int* ix_eq_joint,
        py::capsule t_events,
        T v_0,
        T mu_over_2vs,
        py::capsule K_inner_inner_onfault,
        py::capsule K_inner_asperities_v_plate,
        py::capsule v_plate_ddcs_proj_eff_inner,
        py::capsule v_init,
        T atol,
        T rtol,
        T spinup_atol,
        T spinup_rtol
    )
    {
        // calculate some shapes
        _num_t_eval = t_eval_joint_sec.attr("size").cast<int>();
        _num_ix_eq = ix_eq_joint.attr("size").cast<int>();
        _num_inner_patches = v_plate_ddcs_proj_eff_inner.attr("size").cast<int>() / 2;
        _num_eq = num_eq;

        // check the shapes
        auto temp_int = v_init.attr("size").cast<int>() / 2;
        if (temp_int != _num_inner_patches)
            throw std::runtime_error("num_inner_patches from v_init: "
                                     + std::to_string(_num_inner_patches) + " != " + std::to_string(temp_int));
        temp_int = K_inner_asperities_v_plate.attr("size").cast<int>() / 2;
        if (temp_int != _num_inner_patches)
            throw std::runtime_error("num_inner_patches from K_inner_asperities_v_plate: "
                                     + std::to_string(_num_inner_patches) + " != " + std::to_string(temp_int));
        temp_int = (int) sqrt(K_inner_inner_onfault.attr("size").cast<int>() / 4);
        if (temp_int != _num_inner_patches)
            throw std::runtime_error("num_inner_patches from K_inner_inner_onfault: "
                                     + std::to_string(_num_inner_patches) + " != " + std::to_string(temp_int));

        // initialize CUDA model
        _cmodel->initialize(
            num_systems,
            systems_batch,
            max_cycles,
            _num_t_eval,
            convertPyArray<T, cuda_vector>(t_eval_joint_sec),
            _num_ix_eq,
            _num_eq,
            convertPyArray<int, cuda_vector>(ix_eq_joint),
            convertPyArray<T, cuda_vector>(t_events),
            v_0,
            mu_over_2vs,
            _num_inner_patches,
            convertPyArray<T, cuda_vector>(K_inner_inner_onfault),
            convertPyArray<T, cuda_vector>(K_inner_asperities_v_plate),
            convertPyArray<T, cuda_vector>(v_plate_ddcs_proj_eff_inner),
            convertPyArray<T, cuda_vector>(v_init),
            atol,
            rtol,
            spinup_atol,
            spinup_rtol
        );
    }

    // set system ODEs
    void set_system_odes(py::capsule alpha_h_vec, py::capsule delta_tau_bounded)
    {
        // check some shapes
        auto temp_int = alpha_h_vec.attr("size").cast<int>();
        if (temp_int != _num_systems * _num_inner_patches)
            throw std::runtime_error("Invalid alpha_h_vec size: got "
                                     + std::to_string(_num_systems * _num_inner_patches)
                                     + ", expected " + std::to_string(temp_int));
        temp_int = (int) delta_tau_bounded.attr("size").cast<int>() / _num_systems / _num_inner_patches / 2;
        if (temp_int != _num_eq)
            throw std::runtime_error("num_eq from delta_tau_bounded: "
                                     + std::to_string(_num_eq) + " != " + std::to_string(temp_int));

        // set internal values
        _cmodel->set_system_odes(
            convertPyArray<T, cuda_vector>(alpha_h_vec),
            convertPyArray<T, cuda_vector>(delta_tau_bounded)
        );
    }

    // just pass forward model through
    void forward_model_batch()
    {
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
