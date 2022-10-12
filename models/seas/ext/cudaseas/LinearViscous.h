// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2022 california institute of technology
// all rights reserved
//

// code guard
#if !defined(altar_models_seas_cuda_LinearViscous_h)
#define altar_models_seas_cuda_LinearViscous_h

#include <iostream>

// declaration
namespace altar::models::seas::cuda {
    template<typename T>
    class LinearViscous {
    // methods
    public:
        LinearViscous() {} // default constructor
        ~LinearViscous() {} // default destructor
        // initial parameters and data
        void initialize(int samples_, int patches_, int stations_,
            T t0, T t1, T Vj,
            T* stress_kernel,
            T* stressrate_ext,
            T* displacement_kernel,
            int t_eval_points, T* t_eval,
            T* coseismic,
            int spinup_max_cycles,
            int spinup_convergence_check_cycles);
        // set spin up data (initial
        void set_spinup_data(T* spinup_data);
        // set ode parameters
        void set_ode_parameters(const int steps, const T ta, const T tr);
        // perform forward modeling
        void forward_model(const T* theta, T* prediction, const int parameters, const int batch);
        // f=dy/dt function, forward declaration
        // struct ode_function;


    // parameters
    private:
        int max_samples_;
        int patches_;
        int stations_;


        T* stress_kernel_;
        T* stressrate_ext_;
        T* displacement_kernel_;
        int t_eval_points_;
        T* t_eval_;
        T* coseismic_;    // coseismic change of (slip, velocity)
        T* spinup_data_; // initial value y=(slip, velocity)
        T Vj_;

        T t0_;
        T t1_;

        int spinup_max_cycles_;
        int spinup_convergence_check_cycles_;

        // ode parameters
        int rk_steps_;
        T tolerance_absolute_;
        T tolerance_relative_;

        // temporary data
        T *yold_, *ynew_ ; // save data (slip, velocity) data of a given time (old and new for checking convergence)
        T * y_eval_; // (slip, velocity) data of all t_eval time points
        int * convergence_;

        // private methods
        void add_coseismic_change(T* y, const T* coseismic, const T* alpha1, const int parameters, const int samples, const int patches);
        // call ode solver
        void ode_solver(const int batch, const int parameters, const T* alpha1, const T* y0, T* y1, bool dense_output);
        // set slips to zero, as a temporary solution for convergence
        void set_slips_zero(T* y, const int samples, const int patches);
        bool check_spinup_convergence(const T* yn, const T* yo, const int batch);
        void compute_displacement(const T* yeval, T* predictions, const int batch);
    };
} // end of namespace altar::models::seas::cuda
#endif
