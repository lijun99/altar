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
#include "LinearViscousOde.cuh"

namespace altar::models::seas::cuda {

template <typename T>
void
LinearViscous<T>::ode_solver(const int batch, const int parameters, const T* alpha1,
    const T* yin, T* yout, bool dense_output)
{

    // if not dense_output, we only get the last time point (t1) value
    // otherwise, perform interpolation to get all t_eval time points
    // results are saved in yout
    int nout = (dense_output) ? t_eval_points_ : 1;

    //call the solver
    altar::models::seas::cuda::linearviscous_ode::ode_solver(
        rk_steps_,
        batch,
        2*patches_,
        t0_, // start time
        t1_, // end time
        yin, // initial values for y =(slip, velocity) [samples, 2*patches]
        dense_output,
        t_eval_, // desired output time points [samples, nout]
        yout, // output y values at tout  [samples, nout, 2*patches]
        nout, // number of desired output time points
        Vj_, stress_kernel_,
        parameters, alpha1
        );
    // all done
}

template <typename T>
void
LinearViscous<T>::set_ode_parameters (int steps, T ta, T tr)
{
    rk_steps_ = steps;
    tolerance_absolute_ = ta;
    tolerance_relative_ = tr;
}

// Initialize model parameters
// suffix underline indicate class parameters
template <typename T>
void LinearViscous<T>::initialize(
    int samples, int patches, int stations,
    T t0, T t1, T Vj,
    T* stress_kernel, T* displacement_kernel,
    int t_eval_points, T* t_eval,
    T* coseismic,
    int spinup_max_cycles,
    int spinup_convergence_check_cycles
    )
{
    // assign parameters
    max_samples_ = samples;
    patches_ = patches;
    stations_ = stations;

    stress_kernel_ = stress_kernel;
    displacement_kernel_ = displacement_kernel;
    t_eval_points_ = t_eval_points;
    t_eval_ = t_eval;
    coseismic_ = coseismic;

    t0_ = t0;
    t1_ = t1;
    Vj_ = Vj;

    spinup_max_cycles_ = spinup_max_cycles;
    spinup_convergence_check_cycles_ = spinup_convergence_check_cycles;

    // allocate the (slip, velocity) at t_eval
    cudaSafeCall(cudaMalloc(&y_eval_, t_eval_points_*2*patches_*max_samples_*sizeof(T)));
    cudaSafeCall(cudaMemset(y_eval_, 0, t_eval_points_*patches_*max_samples_*sizeof(T)));

    // alllocate two (old and new) matrix for (slip velocity) at t1
    cudaSafeCall(cudaMalloc(&yold_, 2*patches_*max_samples_*sizeof(T)));
    cudaSafeCall(cudaMalloc(&ynew_, 2*patches_*max_samples_*sizeof(T)));

    // allocate a vector to record convergence
    cudaSafeCall(cudaMalloc(&convergence_, max_samples_*sizeof(int)));

}

template <typename T>
void LinearViscous<T>::set_spinup_data(T* spinup_data)
{
    spinup_data_ = spinup_data;
    std::cout << "assign spin up data";
    details::debug_cuda_memory(spinup_data_, 2*patches_);
}

template <typename T>
void
LinearViscous<T>::forward_model (const T* theta, T* prediction, const int parameters, const int batch)
{
    // load spin up data from preset or previous iteration
    details::matrix_duplicate_vector<T>(ynew_, spinup_data_, batch, 2*patches_);

    // spin up
    int cycles = 0;
    while (cycles < spinup_max_cycles_) {
        for(int cycle=0; cycle< spinup_convergence_check_cycles_; cycle++)
        {
            // make a copy of current state for convergence check
            details::matrix_copy<T>(yold_, ynew_, batch*2*patches_);

            //std::cout << "printing init ynew \n";
            //details::debug_cuda_memory<T>(ynew_, batch*2*patches_);

            // add coseismic change
            add_coseismic_change(ynew_, coseismic_, theta, parameters, batch, patches_);

            //std::cout << "printing theta/alpha1 \n";
            //details::debug_cuda_memory<T>(theta, batch*parameters);

            //std::cout << "printing ynew after adding coseismic \n";
            //details::debug_cuda_memory<T>(ynew_, batch*2*patches_);

            // call ode solver for one cycle, only get the last time point values
            ode_solver(batch, parameters, theta, ynew_, y_eval_, false);

            // copy the last time point values to ynew
            details::matrix_copy<T>(ynew_, y_eval_, batch*2*patches_);

            //std::cout << "printing ynew after ode\n";
            //details::debug_cuda_memory<T>(ynew_, batch*2*patches_);

            // set slip to zeros to enforce convergence
            set_slips_zero(ynew_, batch, patches_);
        }

        // check convergence
        bool converged = check_spinup_convergence(ynew_, yold_, batch);
        if(converged) {
            cycles = spinup_max_cycles_;
        }
        else{
            cycles += spinup_convergence_check_cycles_;
        }
    }

    // save the first set of data to spin up data for later iterations
    details::matrix_copy<T>(spinup_data_, ynew_, 2*patches_);

    // final cycle to compute data output
    // run an ode with dense_output
    add_coseismic_change(ynew_, coseismic_, theta, parameters, batch, patches_);
    ode_solver(parameters, batch, theta, ynew_, y_eval_, true);

    // compute the observations
    compute_displacement(y_eval_, prediction, batch);
    // all done
}


// add coseismic change (slip, stress/alpha1) to y(slip, velocity) - kernel
template<typename T>
__global__ void add_coseismic_change_kernel(T* y,
    const T* coseismic, const T* alpha1,
    const int parameters, const int samples, const int patches)
{
    // each row uses a thread, get row index from thread id
    int sample  = blockIdx.x *blockDim.x + threadIdx.x;
    // avoid out of range
    if (sample>=samples)
        return;
    // get the head of the matrix row
    auto y_s = &y[sample*2*patches];
    auto alpha1_s = &alpha1[sample*parameters];
    // iterate over slip cols
    for(int i=0; i<patches; ++i)
        y_s[i] += coseismic[i];
    // iterate over velocity cols
    for(int i=patches; i<2*patches; ++i)
        y_s[i] += coseismic[i]/alpha1_s[0];
    // all done
}


// add coseismic change (slip, stress/alpha1) to (slip, velocity)
template <typename T>
void LinearViscous<T>::add_coseismic_change(T* y, const T* coseismic, const T* alpha1,
    const int parameters, const int samples, const int patches)
{
    // decide the execution size - one sample per thread
    const int threadsPerBlock = 128;
    const int numberOfBlocks = IDIVUP(samples, threadsPerBlock); //IDIVUP
    // call kernel
    add_coseismic_change_kernel<T><<<numberOfBlocks, threadsPerBlock>>>(y, coseismic, alpha1,
        parameters, samples, patches);
    // check error
    cudaSafeCall(cudaGetLastError());
}


// set slips to 0, as a temporary solution for convergence
template<typename T>
__global__ void set_slips_zero_kernel(T* y, const int samples, const int patches)
{
    // each row uses a thread, get row index from thread id
    int sample  = blockIdx.x *blockDim.x + threadIdx.x;
    // avoid out of range
    if (sample>=samples)
        return;
    // get the head of the matrix row
    auto y_s = y + sample*2*patches;

    for(int i=0; i<patches; ++i)
        y_s[i] = 0;
    // all done
}

template <typename T>
void LinearViscous<T>::set_slips_zero(T* y, const int samples, const int patches)
{
    // decide the execution size - one sample per thread
    const int threadsPerBlock = 128;
    const int numberOfBlocks = (samples-1+threadsPerBlock)/threadsPerBlock; //IDIVUP
    // call kernel
    set_slips_zero_kernel<T><<<numberOfBlocks, threadsPerBlock>>>(y, samples, patches);
    // check error
    cudaSafeCall(cudaGetLastError());
}

template<typename T>
__device__
int check_error(const T a, const T b, const T absolute, const T relative)
{
    auto diff = abs(a-b);
    if( diff > absolute)
        return 1;
    else if (diff/(abs(b)+absolute) > relative)
        return 1;
    else
        return 0;
}

template<typename T>
__global__ void check_spinup_convergence_kernel(int* result, const T* y0, const T* y1,
    const T absolute, const T relative, const int samples, const int elements)
{
    // each row uses a thread, get row index from thread id
    int sample  = blockIdx.x *blockDim.x + threadIdx.x;
    // avoid out of range
    if (sample>=samples)
        return;
    // get the head of the matrix row
    auto y0s = y0 + sample*elements;
    auto y1s = y1 + sample*elements;

    result[sample] = 0;
    for(int i=0; i<elements; ++i)
        result[sample] += check_error<T>(y0s[i], y1s[i], absolute, relative);
    // all done
}

template <typename T>
bool LinearViscous<T>::check_spinup_convergence(const T* y0, const T* y1, const int samples)
{
    // decide the execution size - one sample per thread
    const int threadsPerBlock = 128;
    const int numberOfBlocks = (samples-1+threadsPerBlock)/threadsPerBlock; //IDIVUP
    // call kernel
    check_spinup_convergence_kernel<T><<<numberOfBlocks, threadsPerBlock>>>(convergence_, y0, y1,
        tolerance_absolute_, tolerance_relative_, samples, 2*patches_);
    // check error
    cudaSafeCall(cudaGetLastError());
    // convergence_ records convergence for each sample
    // note: currently, if one sample is not converged, we repeat for all
    int sum = details::vector_sum<int>(convergence_, samples);
    return sum==0;
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

}

// explicit instantiation
template class altar::models::seas::cuda::LinearViscous<float>;
template class altar::models::seas::cuda::LinearViscous<double>;

} // end of namespace
// end of file
