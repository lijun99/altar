/**
 * pyLinearViscous.cc - python wrapper for Linear Viscous ODE Solver
 **/

// namespace setup
#include "external.h"
#include "forward.h"


// my definitions
#include "LinearViscous.h"


namespace altar::cuda::py::seas::linearviscous {

// define a struct as interface to call c modules
template<typename T>
struct ode_solver {
    // fixed parameters, initialized in the beginning
    int system_size;     // 2*patches (slip, velocity)
    int asperity_range;  // creep zone
    T Vj;                // backslip
    T* stressKernel;     // stress kernel

    // methods - we need wrapper to convert python objects to c/c++ objects
    // current cuda matrix/vectors are passed by python capsules
    // this will be changed when all modules are implemented by pybind11
    void init(int ss, int a_r, double v, py::capsule s);
    void run(
        int samples,
        T t0, // start time
        T tn, // end time
        int rk_steps, // number of rk steps between t0 and tn
        py::capsule py_y0, // initial values for y =(slip, velocity) [samples, 2*patches]
        bool dense_output,
        py::capsule py_tout, // desired output time points [samples, nout]
        py::capsule py_yout, // output y values at tout  [samples, nout*2*patches]
        int nout, // number of desired output time points
        py::capsule py_alpha1 // viscous coefficient, a constant for all patches in each sample [samples]
        );
};

template<typename T>
void ode_solver<T>::init(int ss, int a_r, double v, py::capsule s)
{
    // assign values
    system_size = ss;
    asperity_range = a_r;
    Vj = (T)v;
    // cast python matrix to C matrix
    cuda_matrix* s_mat = static_cast<cuda_matrix *>(s.get_pointer());
    // get the data pointer
    stressKernel = (T *)s_mat->data;
}

template<typename T>
void ode_solver<T>::run(
        int samples,
        T t0, // start time
        T tn, // end time
        int rk_steps, // number of rk steps between t0 and tn
        py::capsule py_y0, // initial values for y =(slip, velocity) [samples, 2*patches]
        bool dense_output,
        py::capsule py_tout, // desired output time points [samples, nout]
        py::capsule py_yout, // output y values at tout  [samples, nout, 2*patches]
        int nout, // number of desired output time points
        py::capsule py_alpha1 // viscous coefficient, a constant for all patches in each sample [samples]
        )
{
    // cast python matrix/vectors
    auto y0_c = static_cast<cuda_vector *>(py_y0.get_pointer());
    T* y0 = (T *)y0_c->data;
    auto tout_c = static_cast<cuda_vector *>(py_tout.get_pointer());
    T* tout = (T *)tout_c->data;
    auto yout_c = static_cast<cuda_matrix *>(py_yout.get_pointer());
    T* yout = (T *)yout_c->data;
    auto alpha1_c = static_cast<cuda_matrix *>(py_alpha1.get_pointer());
    T* alpha1 = (T *)alpha1_c;
    // call c method
    altar::models::seas::cuda::linearviscous::ode_solver<T>(
        samples, system_size, t0, tn, rk_steps, y0, dense_output, tout, yout, nout,
        asperity_range, Vj, stressKernel, alpha1);
    // all done
}



// add bindings for the various cuda struct
void
module(py::module & m)
{
    using ode_solver_double = ode_solver<double>;
    using ode_solver_float = ode_solver<float>;

    py::class_<ode_solver_double>(m, "ode_solver_double")
        .def(py::init())
        .def("init", &ode_solver_double::init)
        .def("run", &ode_solver_double::run);
    py::class_<ode_solver_float>(m, "ode_solver_float")
        .def(py::init())
        .def("init", &ode_solver_float::init)
        .def("run", &ode_solver_float::run);

}

} // end of namespace pycuda::seas::linearviscous

// end of file
