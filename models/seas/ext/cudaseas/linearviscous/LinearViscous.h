// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2022 california institute of technology
// all rights reserved
//

// code guard
#if !defined(altar_models_seas_cuda_LinearViscous_h)
#define altar_models_seas_cuda_LinearViscous_h

// cuda ode solver
#include <cudaode.cuh>
// my definitions of ode and events (coseismic)
#include "Ode.cuh"
#include "Events.cuh"


// declaration
namespace altar::models::seas::cuda::linearviscous {

template<typename T>
class LinearViscous {
// methods
public:

    // types
    using OdeType = Ode<T>; // ode function defition from Ode.cuh
    using EventType = Events<T>; // event(coseismic) from Events.cuh
    // this is an ode solver with spin up procedure built-in
    using SolverType = cuda::ode::dopri5::SpinupSolver<T, OdeType, EventType>;

    LinearViscous() {}; // default constructor
    ~LinearViscous() {} // default destructor

    // initial parameters and data
    void initialize(int samples_, int patches_, int stations_,
        T t0, T t1, T Vj,
        T* stress_kernel,
        T* stressrate_ext,
        T* displacement_kernel,
        int t_eval_points, T* t_eval,
        int n_coseismic, T* t_cosemisc, T* coseismic,
        int spinup_max_cycles,
        int spinup_convergence_check_cycles);

    // set spin up data (initial values)
    void set_spinup_data(T* spinup_data);

    // perform forward modeling
    void forward_model(const T* theta, T* prediction, const int parameters, const int batch);


// parameters
private:

    OdeType * odefunc;
    EventType * events;
    SolverType * solver;

    int max_samples;
    int patches;
    int system_size;
    int stations;

    // rheology
    T Vj;
    T* stress_kernel;
    T* stressrate_ext;
    T* displacement_kernel;

    // events
    int n_coseismic; // nevents
    T * t_coseismic;
    T* coseismic;    // coseismic change of (slip, velocity)

    // y0
    T* y0; // initial value y=(slip, velocity)

    // output of ode
    int neval;
    const T* teval; // input
    T* yeval; // input

    // ode control parameters
    T atol;
    T rtol;
    int spinup_max_cycles;


    // set slips to zero, as a temporary solution for convergence

}; //end of struct LinearViscous

} // end of namespace altar::models::seas::cuda
#endif
