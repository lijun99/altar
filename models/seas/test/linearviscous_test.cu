#include <iostream>
#include "LinearViscousOde.cuh"

int main()
{
    using data_type = float;


    const int size = 128;
    auto patches = size/2;
    const int samples = 64;

    data_type *y0;
    data_type t0 = 0, tn = 1;

    const int steps = 1024;
    const int nout = 32+1;
    data_type * tout, *yout;
    data_type *alpha ;
    data_type *stress;
    data_type vj=1.0;

    cudaSafeCall(cudaMallocManaged(&y0, samples*size*sizeof(data_type)));
    cudaSafeCall(cudaMallocManaged(&tout, nout*sizeof(data_type)));
    cudaSafeCall(cudaMallocManaged(&yout, samples*nout*size*sizeof(data_type)));
    cudaSafeCall(cudaMallocManaged(&stress, patches*patches*sizeof(data_type)));
    cudaSafeCall(cudaMallocManaged(&alpha, samples*sizeof(data_type)));

    cudaSafeCall(cudaMemset(y0, 0, samples*size*sizeof(data_type)));
    cudaSafeCall(cudaMemset(stress, 0, patches*patches*sizeof(data_type)));

    for(int i=0; i<samples; i++)
        alpha[i] = 0.1;

    for(int i=0; i<nout; i++)
        tout[i] = (i+1)/nout;

    for(int i=0; i<3; i++) {
        altar::models::seas::cuda::linearviscous_ode::ode_solver<data_type>
            (steps, samples, size, t0, tn, y0, 1, tout, yout, nout, vj, stress, 1, alpha);
        cudaSafeCall(cudaDeviceSynchronize());
    }
    cudaDeviceSynchronize();

    for(int i=0; i< patches; i++ )
        std::cout << yout[i] << " ";
    std::cout << "\n";

}
