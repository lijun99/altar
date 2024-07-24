// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2022 - 2023 california institute of technology
// all rights reserved

// an example with ode defined in myode and events/time defined in events
// compile with, e.g., nvcc -arch=native example.cu -I../../

#include <cstdio>
#include <iostream>
#include <iomanip>
// dopri5 solver
#include "cudaode.cuh"
// my ode and event definitions
#include "myode.cuh"
#include "myevent.cuh"

int main()
{
    // some type renames
    using T=float;
    using EventType = MyTestEvents<T>;
    using OdeType = MyTestOde<T>;
    using SolverType = cuda::ode::dopri5::Solver<T, OdeType, EventType>;

    // define ode system
    const int patches = 1024;
    const int units = 2;
    const int systems = 1024;
    const int systems_batch = 1024;
    const int threads = 1024;
    const int system_size = patches*units;
    OdeType odefunc {patches, units, systems};

    const T t0 = 0;
    const T t1 = 1;

    // input, use the same y0 for all systems
    T *y0;
    cudaMallocManaged(&y0, system_size*sizeof(T));
    for(auto i=0; i<patches*units; i++)
        y0[i] = static_cast<T>(1.0);

    // set dense output
    const int neval = 5;
    T * teval;
    T * yeval;
    cudaMallocManaged(&teval, neval*sizeof(T));
    auto size_yeval = (size_t)systems*neval*system_size*sizeof(T);
    cudaMallocManaged(&yeval, size_yeval);
    for(auto i=0; i<neval; i++)
        teval[i] = t0+i*(t1-t0)/(neval-1);

    // set myevents
    const int nevents = 3;
    EventType events {nevents, t0, t1, patches*units};

    bool dense_out = true;
    bool use_y0_for_all = true;
    const T atol = 1e-10;
    const T rtol = 1e-8;

    // construct the ode solver
    SolverType solver(odefunc, events, atol, rtol, systems_batch, threads);
    // set dense output
    solver.set_dense_output(neval, teval, yeval);
    // iteratively solve systems in batch
    for(int system_offset =0; system_offset<systems; system_offset+=systems_batch)
    {
        // check how many systems are left
        auto systems_to_process = min(systems_batch, systems-system_offset);

        std::cout << "running systems " << system_offset << " to " << system_offset+systems_to_process << "\n";
        // set initial values
        solver.set_init_values(y0, use_y0_for_all, systems_to_process, system_offset);
        std::cout << "setting initial values done " << "\n";
        // call the solver
        solver.solve_ivp(dense_out, systems_to_process, system_offset);
        std::cout << "solving ivp done " << "\n";
        cudaDeviceSynchronize();
        for(auto is = max(0, systems_to_process-4); is<systems_to_process; is++)
        {
            auto system = is + system_offset;
            std::cout << "System " << system << " ... \n";
            for(auto ieval = 0; ieval<neval; ieval++)
            {
                std::cout << "t=" << std::setw(8) << teval[ieval] << " ";
                for(auto i=0; i<min(8, patches*units); i++)
                    std::cout << std::setw(8) << yeval[(system*neval+ieval)*patches*units+i] << " ";
                std::cout << "...\n";
            }
        }
    }


    // check the results
    cudaDeviceSynchronize();


    //all done
}
// end of file
