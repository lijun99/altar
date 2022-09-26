#include <iostream>
#include "LinearViscousOde.cuh"

int main()
{
    using data_type = double;


    const int size = 1024;
    auto patches = size/2;
    const int samples = 256;

    data_type *y0;
    data_type t0 = 0, tn = 1;

    const int steps = 1024;
    const int nout = 32+1;
    data_type * tout, *yout;
    data_type *alpha ;
    data_type *stress;
    data_type vj=-1.0;

    cudaMallocManaged(&y0, samples*size*sizeof(data_type));
    cudaMallocManaged(&tout, nout*sizeof(data_type));
    cudaMallocManaged(&yout, samples*nout*size*sizeof(data_type));
    cudaMallocManaged(&stress, patches*patches*sizeof(data_type));
    cudaMallocManaged(&alpha, samples*sizeof(data_type));

    cudaMemset(y0, 0, samples*size*sizeof(data_type) );
    cudaMemset(stress, 0, samples*size*nout*sizeof(data_type) );

    for(int i=0; i<samples; i++)
        alpha[i] = 0.1;

    for(int i=0; i<nout; i++)
        tout[i] = (i+1)/nout;

    altar::models::seas::cuda::linearviscous_ode::ode_solver<double>
        (steps, samples, size, t0, tn, y0, 1, tout, yout, nout, vj, stress, 1, alpha);

    cudaDeviceSynchronize();

    for(int i=0; i< patches; i++ )
        std::cout << yout[i] << " ";
    std::cout << "\n";

}
