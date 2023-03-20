/**
 * pyLinearViscous.cc - python wrapper for Linear Viscous ODE Solver
 **/

// namespace setup
#include "external.h"
#include "forward.h"


// my definitions
#include "LinearViscous.h"


namespace altar::cuda::py::seas::linearviscous {

template <typename T, typename D>
T* convertPyArray(py::capsule pycap)
{
    auto cap = static_cast<D *>(pycap.get_pointer());
    return (T *)cap->data;
}


template<typename T>
class pyLinearViscous
{
    using model_type = altar::models::seas::cuda::LinearViscous<T>;
private:
    model_type * _cmodel;
public:
    // constructor
    pyLinearViscous () {_cmodel = new model_type();}
    // initialize cmodel parameters
    void initialize(int samples, int patches, int stations,
        T t0, T t1, T Vj,
        py::capsule stress_kernel,
        py::capsule stressrate_ext,
        py::capsule displacement_kernel,
        int t_eval_points, py::capsule t_eval,
        int n_coseismic, py::capsule t_coseismic, py::capsule coseismic,
        int spin_up_max_cycles,
        int spin_up_convergence_check_cycles)
    {
        _cmodel->initialize(
            samples, patches, stations,
            t0, t1, Vj,
            convertPyArray<T, cuda_matrix>(stress_kernel),
            convertPyArray<T, cuda_vector>(stressrate_ext),
            convertPyArray<T, cuda_matrix>(displacement_kernel),
            t_eval_points, convertPyArray<T, cuda_vector>(t_eval),
            n_coseismic, convertPyArray<T, cuda_vector>(t_coseismic),
            convertPyArray<T, cuda_vector>(coseismic),
            spin_up_max_cycles, spin_up_convergence_check_cycles
        );
    }
    // set initial spin up state
    void set_spinup_data(py::capsule spinup_data)
    {
        _cmodel->set_spinup_data(
            convertPyArray<T, cuda_vector>(spinup_data)
        );
    }
    // set ode solver parameters
    void set_ode_parameters(int steps, T tolerance_absolute, T tolerance_relative)
    {
        _cmodel->set_ode_parameters(steps, tolerance_absolute, tolerance_relative);
    }
    // forward modeling theta -> data prediction
    void forward_model(py::capsule theta, py::capsule prediction, int parameters, int batch)
    {
        auto c_theta =  convertPyArray<T, cuda_matrix>(theta);
        auto c_prediction = convertPyArray<T, cuda_matrix>(prediction);
        _cmodel->forward_model( c_theta, c_prediction, parameters, batch);
    }
};

// add bindings for the various cuda struct
void
module(py::module & m)
{
    using pyLinearViscous_float = pyLinearViscous<float>;
    using pyLinearViscous_double = pyLinearViscous<double>;

    py::class_<pyLinearViscous_double>(m, "model_double")
        .def(py::init())
        .def("initialize", &pyLinearViscous_double::initialize)
        .def("set_spinup_data", &pyLinearViscous_double::set_spinup_data)
        .def("set_ode_parameters", &pyLinearViscous_double::set_ode_parameters)
        .def("forward_model", &pyLinearViscous_double::forward_model);
    py::class_<pyLinearViscous_float>(m, "model_float")
        .def(py::init())
        .def("initialize", &pyLinearViscous_float::initialize)
        .def("set_spinup_data", &pyLinearViscous_float::set_spinup_data)
        .def("set_ode_parameters", &pyLinearViscous_float::set_ode_parameters)
        .def("forward_model", &pyLinearViscous_float::forward_model);
}

} // end of namespace pycuda::seas::linearviscous

// end of file
