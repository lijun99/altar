// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2022 california institute of technology
// all rights reserved
//

// for the build system
#include <portinfo>

// get my class declaration
#include "LinearViscous.h"

// get my dependencies
#include "dopri5log.cuh"

namespace altar::models::seas::cuda {

// Initialize model parameters
// suffix underline indicate class parameters
template <typename T>
void LinearViscous<T>::initialize(
    int samples, int patches, int stations,
    T Vj, T* stress_kernel, T* stressrate_ext,
    T* displacement_kernel,
    int nevent, const T* tevent, const T* yevent,
    int t_eval_points, T* t_eval,
    int n_coseismic, T* t_coseismic, T* coseismic,
    T atol, T rtol, int spinup_max_cycles,
    )
{
    // assign parameters
    max_samples = samples;
    patches_ = patches;
    stations_ = stations;

    // ode
    Vj = Vj_;
    stress_kernel = stress_kernel_;
    stressrate_ext = stressrate_ext_;


    // create an instance of odefunc
    odefunc = new OdeType(max_samples, patches, 2);
    odefunct->initialize(Vj, stress_kernel, stressrate_ext);

    // create an instance of events (including starting/ending time)
    events = new EventType(nevents_, tevents_, coseismic_; patches*2);

    // create the solver
    solver = new SolverType(*odefunc, *events, atol_, rtol_, max_samples);

    // output
    neval = neval_;
    teval = teval_;
    yeval = yeval_;
    solver->set_dense_output(neval, teval, yeval);

    spinup_max_cycles = spinup_max_cycles_;
}


template <typename T>
void
LinearViscous<T>::forward_model (const T* theta, T* prediction, const int parameters, const int batch)
{
    // set initial values
    // parameters y0, use_y0_for_all, systems_to_process, system_offset
    solver->set_init_values(y0, true, batch, 0);

    // set parameters (pointer) to odefunc
    odefunc.set_alpha1(theta, parameters);

    // iteratively solve ode until convergence
    // paramtgers (dense_out, systems_to_process, system_offset, max_cycles)
    solver-> solve_ivp_cycles(true, batch, 0, spinup_max_cycles);
    // results are saved in yeval

    // compute the observations
    compute_displacement(yeval, prediction, batch);
    // all done
}


template<typename T>
__global__
void compute_displacement_kernel(const T* yeval, const T* gf, T* predictions,
        const int samples, const int t_points, const int patches, const int stations)
{
    // each row uses a thread, get row index from thread id
    int sample  = blockIdx.x *blockDim.x + threadIdx.x;
    // avoid out of range
    if (sample>=samples)
        return;

    // get the head of the matrix for this sample
    auto y_s = yeval + sample*t_points*2*patches;
    auto pred_s = predictions + sample*t_points*stations;

    // iterate over time points
    for(int t=0; t<t_points; t++)
    {
        // get data pointers for this sample at this time
        auto y_s_t = y_s + t*2*patches;
        auto pred_s_t = pred_s +t*stations;
        auto gf_t = gf + t*patches*stations;

        // compute Obs (stations) = Slip (patches) x G(patches, stations)
        for(int s=0; s<stations; s++)
        {
            pred_s_t[s] = 0;
            for (int p=0; p<patches; p++)
                pred_s_t[s] += y_s_t[p]*gf_t[p*stations+s];
        }

    // printf("disp %d %d %d %g %g %g\n", sample, samples, t, pred_s_t[0], y_s_t[0], gf_t[0]);
    }
}

// compute displacement from slips
// @note yeval is arranged in shape (samples, times, 2*patches) - C-style
//       gf is arranged in shape (times, patches, stations)
//          - merged with data covariance, therefore, different for different t
//       predictions is arranged in shape (samples, times, stations)
template <typename T>
void LinearViscous<T>::compute_displacement(const T* yeval, T* predictions, const int samples)
{
        // decide the execution size - one sample per thread
    const int threadsPerBlock = 128;
    const int numberOfBlocks = (samples-1+threadsPerBlock)/threadsPerBlock; //IDIVUP
    // call kernel
    compute_displacement_kernel<T><<<numberOfBlocks, threadsPerBlock>>>(yeval, displacement_kernel_,predictions,
        samples, t_eval_points_,patches_,stations_);
    // check error
    cudaSafeCall(cudaGetLastError());

    // std::cout << "printing predictions after ode \n";
    //details::debug_cuda_memory<T>(displacement_kernel_, t_eval_points_*stations_*2*patches_);
    // details::debug_cuda_memory<T>(predictions, samples*t_eval_points_*stations_);

}

// explicit instantiation
template class altar::models::seas::cuda::LinearViscous<float>;
template class altar::models::seas::cuda::LinearViscous<double>;

} // end of namespace
// end of file
