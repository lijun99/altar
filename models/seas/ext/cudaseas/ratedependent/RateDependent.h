// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2022 california institute of technology
// all rights reserved
//

// code guard
#if !defined(altar_models_seas_cuda_RateDependent_h)
#define altar_models_seas_cuda_RateDependent_h

// cuda ode solver
#include "cudaode.cuh"
// my definitions of ode and events (coseismic)
#include "Ode.cuh"
#include "Events.cuh"


// declaration
namespace altar::models::seas::cuda::ratedependent {

template<typename T>
class RateDependent {
// methods
public:

    // types
    using OdeType = Ode<T>; // ode function defition from Ode.cuh
    using EventType = Events<T>; // event(coseismic) from Events.cuh
    // this is an ode solver with spin up procedure built-in
    using SolverType = ::cuda::ode::dopri5::SpinupSolver<T, OdeType, EventType>;

    RateDependent() = default; // default constructor
    ~RateDependent() = default; // default destructor

    // initial parameters and data
    void initialize(
        int max_samples_, int patches_, int stations_, //
        T Vj_, T mu_over_2vs_,
        T* stress_kernel_, // patches * patches
        T* stressrate_ext_, // patches
        T* displacement_kernel_, //
        int n_coseismic_, T* t_coseismic_, T* coseismic_, // events
        int neval_, T* teval_, T* yeval_,
        T atol_, T rtol_, int spinup_max_cycles_ // ode controls
        );

    // set spin up data (initial values)
    void set_initial_values(T* spinup_data) { y0 = spinup_data; };

    // compute displacement
    void compute_displacement(const T* yeval, T* predictions, const int batch);

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
    T mu_over_2vs;
    T* stress_kernel;
    T* stressrate_ext;
    T* displacement_kernel;

    // events
    int n_coseismic; // nevents
    T * t_coseismic;
    T* coseismic;    // coseismic change of (slip, log velocity)

    // y0
    T* y0; // initial value y=(slip, log velocity)

    // output of ode
    int neval;
    const T* teval; // input
    T* yeval; // input

    // ode control parameters
    T atol;
    T rtol;
    int spinup_max_cycles;

}; //end of class RateDependent

} // end of namespace

#endif
// end of file