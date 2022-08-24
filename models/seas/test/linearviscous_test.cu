
int main()
{
    using data_type = double;


    const int size = 256;
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

    altar::models::seas::cuda::linearviscous::ode_solver<double>(samples, size, t0, tn, steps, y0, 1, tout, yout, nout, patches, vj, stress, alpha);

}
