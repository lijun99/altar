#include <iostream>

#include "../dopri5.cuh"

template <typename T>
struct dydt {
    T alpha1;
    __device__ __host__ void operator() (
    T* f, const T t, const T *y0, const int size, const T alpha, const T* alpha2)
    {
        for(int i=0; i<size; ++i)
            f[i] = y0[i]*exp(-alpha*t)/alpha2[i];
    }
};

template <typename T>
void run(const int samples,
        const int size,
        const T t0, const T tn,
        const int steps,
        const T* y0,
        const bool dense,
        const T* tout,
        T* yout,
        const int nout,
        const T alpha,
        const T* alpha2)
{
    dydt<T> f;
    f.alpha1 = 4.0;

    ode::dopri5::rk_solver_fixedstep_batch<T, dydt<T>, const T, const T*><<<1, 256>>>(
        samples,
        size,
        t0, tn, steps,
        y0,
        dense,
        tout, yout, nout,
        f, alpha, alpha2);

    auto status = cudaGetLastError();
    if(status != cudaSuccess)
        std::cout << cudaGetErrorString(status);

    cudaDeviceSynchronize();
    status = cudaGetLastError();
    if(status != cudaSuccess)
        std::cout << cudaGetErrorString(status);
}

template void run<double>(const int samples,
        const int size,
        const double t0, const double tn,
        const int steps,
        const double* y0,
        const bool dense,
        const double* tout,
        double* yout,
        const int nout,
        const double alpha,
        const double* alpha2);

int main()
{
    using data_type = double;
    data_type alpha = 2.0;
    const int size = 256;
    const int samples = 256;

    data_type *y0;
    data_type t0 = 0, tn = 1;

    const int steps = 1024;
    const int nout = 32+1;
    data_type * tout, *yout;

    cudaMallocManaged(&y0, samples*size*sizeof(data_type));
    cudaMallocManaged(&tout, nout*sizeof(data_type));
    cudaMallocManaged(&yout, samples*nout*size*sizeof(data_type));

    auto status = cudaGetLastError();
    if(status != cudaSuccess)
        std::cout << cudaGetErrorString(status);


    for(int s=0; s<samples; s++) {
        for(int i=0; i<size; ++i)
            y0[i+s*size] = i+1;

    }

    for(int i=0; i<nout; ++i)
            tout[i] = (tn-t0)*i/(nout-1);

    run<data_type>(samples,
        size,
        t0, tn, steps,
        y0,
        1,
        tout, yout, nout,
        alpha, y0);

    for(int i=0; i<nout; ++i)
        std::cout << tout[i] << " " << yout[i*size+3*size*nout] << " " << yout[i*size+(samples-1)*size*nout] << " "<<  "\n ";

    std::cout << "\n";

}
