/**
 * pyLinearViscous.cc - python wrapper for Linear Viscous ODE Solver
 **/

// namespace setup
#include "external.h"
#include "forward.h"


// my definitions
#include "linearviscous/LinearViscous.h"
#include "statistics.h"


namespace altar::cuda::py::seas::linearviscous {

template<typename T, class MethodType = ::cuda::ode::dopri5::Dopri5<T>>
class pyLinearViscous
{
    using model_type = altar::models::seas::cuda::linearviscous::LinearViscous<T, MethodType>;
private:
    model_type * _cmodel;
public:
    // constructor
    pyLinearViscous () {_cmodel = new model_type();}

    // the step statistics of the last batch
    py::dict step_statistics() { return statistics_dict(_cmodel->statistics); }
    // initialize cmodel parameters
    void initialize(int samples, int patches, int stations,
        T Vj,
        grid_t & stress_kernel,
        grid_t & stressrate_ext,
        grid_t & displacement_kernel,
        int n_coseismic, grid_t & t_coseismic, grid_t & coseismic,
        int t_eval_points, grid_t & t_eval, grid_t & y_eval,
        T atol, T rtol, T spinup_atol, T spinup_rtol, int spin_up_max_cycles)
    {
        _cmodel->initialize(
            samples, patches, stations,
            Vj,
            cells<T>(stress_kernel),
            cells<T>(stressrate_ext),
            cells<T>(displacement_kernel),
            n_coseismic, cells<T>(t_coseismic),
            cells<T>(coseismic),
            t_eval_points, cells<T>(t_eval),
            cells<T>(y_eval),
            atol, rtol, spinup_atol, spinup_rtol, spin_up_max_cycles
        );
    }
    // set initial spin up state
    void set_spinup_data(grid_t & spinup_data)
    {
        _cmodel->set_initial_values(
            cells<T>(spinup_data)
        );
    }

    void forward_model(grid_t & theta, grid_t & prediction, int parameters, int batch)
    {
        auto c_theta =  cells<T>(theta);
        auto c_prediction = cells<T>(prediction);
        _cmodel->forward_model( c_theta, c_prediction, parameters, batch);
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
        .def("initialize", &P::initialize)
        .def("set_spinup_data", &P::set_spinup_data)
        .def("forward_model", &P::forward_model);
}

// add bindings for the various cuda struct, integrating with dopri5 or radau5
void
module(py::module & m)
{
    using ::cuda::ode::radau5::Radau5;
    bind<pyLinearViscous<double>>(m, "model_double");
    bind<pyLinearViscous<float>>(m, "model_float");
    bind<pyLinearViscous<double, Radau5<double>>>(m, "model_double_radau5");
    bind<pyLinearViscous<float, Radau5<float>>>(m, "model_float_radau5");
}

} // end of namespace pycuda::seas::linearviscous

// end of file
