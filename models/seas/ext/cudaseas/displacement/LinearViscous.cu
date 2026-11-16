// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2022 california institute of technology
// all rights reserved
//

// for the build system
#include <portinfo>

namespace altar::models::seas::cuda::displacement {

// Initialize model parameters
// suffix underline indicate class parameters
template <typename T>
void LinearViscous<T>::initialize(
    int max_samples_, int patches_, int stations_, //
    T Vj_,
    T* stress_kernel_, // patches * patches
    T* stressrate_ext_, // patches
    T* displacement_kernel_, //
    int n_coseismic_, T* t_coseismic_, T* coseismic_, // events
    int neval_, T* teval_, T* yeval_,
    T atol_, T rtol_, int spinup_max_cycles_ // ode controls
    )
{
    // assign parameters
    max_samples = max_samples_;
    patches = patches_;
    system_size = patches*2; //units = 2, s and v
    stations = stations_;

    // ode
    Vj = Vj_;
    stress_kernel = stress_kernel_;
    stressrate_ext = stressrate_ext_;

    // create an instance of odefunc
    odefunc = new OdeType(max_samples, patches, 2);
    odefunc->init_parameters(Vj, stress_kernel, stressrate_ext);

    // create an instance of events (including starting/ending time)
    events = new EventType(n_coseismic_, t_coseismic_, coseismic_, patches*2);

    // create the solver
    solver = new SolverType(*odefunc, *events, atol_, rtol_, max_samples);

    // output
    neval = neval_;
    teval = teval_;
    yeval = yeval_;
    solver->set_dense_output(neval_, teval_, yeval_);

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
    odefunc->set_alpha1(theta, parameters);

    // iteratively solve ode until convergence
    // parameters (dense_out, systems_to_process, system_offset, max_cycles)
    solver-> solve_ivp_cycles(true, batch, 0, spinup_max_cycles);
    // results are saved in yeval

    // compute the observations
    compute_displacement(yeval, prediction, batch);
    // all done
}

// need to rewrite to use block per system
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
    for(int it=0; it<t_points; it++)
    {
        // get data pointers for this sample at this time
        auto y_s_t = y_s + it*2*patches;
        auto pred_s_t = pred_s +it*stations;
        auto gf_t = gf + it*patches*stations;

        // compute Obs (stations) = Slip (patches) x G(patches, stations)
        // will use device blas function to optimize
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
    compute_displacement_kernel<T><<<numberOfBlocks, threadsPerBlock>>>(yeval,
        displacement_kernel, predictions,
        samples, neval, patches, stations);
    // check error
    // cudaSafeCall(cudaGetLastError());
}

// explicit instantiation
template class altar::models::seas::cuda::linearviscous::LinearViscous<float>;
template class altar::models::seas::cuda::linearviscous::LinearViscous<double>;

} // end of namespace
// end of file
