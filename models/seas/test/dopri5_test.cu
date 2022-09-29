#include <iostream>

#include "dopri5.cuh"

template <typename T>
struct dydt {
    __device__ __host__ void operator() (
    T* f, const T t, const T *y0, const int size, const T alpha)
    {
        for(int i=0; i<size; ++i)
            f[i] = y0[i]*exp(-alpha*t);
    }
};


template <typename T>
void run(
        const int steps,
        const int samples,
        const int size,
        const T t0, const T tn,
        const T* y0,
        const bool dense,
        const T* tout,
        T* yout,
        const int nout,
        const T alpha)
{
    dydt<T> f;

    const int threads = 256;
    const int blocks = (samples-1+threads)/threads;
    cudaSafeCall(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 256*1024*1024));
    std::cout << "blocks " << blocks << "\n";

    ode::dopri5::rk_solver_fixedstep_batch<T, dydt<T>, const T><<<blocks, threads>>>(
        steps,
        samples,
        size,
        t0, tn,
        y0,
        dense,
        tout, yout, nout,
        f, alpha);


}


int main()
{
    using data_type = float;
    data_type alpha = 2.0;
    const int size = 256;
    const int samples = 1024;

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

    run<data_type>(steps, samples,
        size,
        t0, tn,
        y0,
        1,
        tout, yout, nout,
        alpha);
    status = cudaGetLastError();
    if(status != cudaSuccess)
        std::cout << cudaGetErrorString(status);

    cudaDeviceSynchronize();
    status = cudaGetLastError();
    if(status != cudaSuccess)
        std::cout << cudaGetErrorString(status);

    for(int i=0; i<nout; ++i)
        std::cout << tout[i] << " " << yout[i*size+3*size*nout] << " " << yout[i*size+(samples-1)*size*nout] << " "<<  "\n ";

    std::cout << "\n";

}
