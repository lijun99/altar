/**
 * pyLinearViscous.cc - python wrapper for Linear Viscous ODE Solver
 **/

// namespace setup
#include "external.h"
#include "forward.h"


// my definitions
#include "linearviscous/LinearViscous.h"


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
    using model_type = altar::models::seas::cuda::linearviscous::LinearViscous<T>;
private:
    model_type * _cmodel;
public:
    // constructor
    pyLinearViscous () {_cmodel = new model_type();}
    // initialize cmodel parameters
    void initialize(int samples, int patches, int stations,
        T Vj,
        py::capsule stress_kernel,
        py::capsule stressrate_ext,
        py::capsule displacement_kernel,
        int n_coseismic, py::capsule t_coseismic, py::capsule coseismic,
        int t_eval_points, py::capsule t_eval, py::capsule y_eval,
        T atol, T rtol, int spin_up_max_cycles)
    {
        _cmodel->initialize(
            samples, patches, stations,
            Vj,
            convertPyArray<T, cuda_matrix>(stress_kernel),
            convertPyArray<T, cuda_vector>(stressrate_ext),
            convertPyArray<T, cuda_matrix>(displacement_kernel),
            n_coseismic, convertPyArray<T, cuda_vector>(t_coseismic),
            convertPyArray<T, cuda_vector>(coseismic),
            t_eval_points, convertPyArray<T, cuda_vector>(t_eval),
            convertPyArray<T, cuda_matrix>(y_eval),
            atol, rtol, spin_up_max_cycles
        );
    }
    // set initial spin up state
    void set_spinup_data(py::capsule spinup_data)
    {
        _cmodel->set_initial_values(
            convertPyArray<T, cuda_vector>(spinup_data)
        );
    }

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
        .def("forward_model", &pyLinearViscous_double::forward_model);
    py::class_<pyLinearViscous_float>(m, "model_float")
        .def(py::init())
        .def("initialize", &pyLinearViscous_float::initialize)
        .def("set_spinup_data", &pyLinearViscous_float::set_spinup_data)
        .def("forward_model", &pyLinearViscous_float::forward_model);
}

} // end of namespace pycuda::seas::linearviscous

// end of file
