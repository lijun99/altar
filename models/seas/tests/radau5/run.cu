// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2026 california institute of technology
// all rights reserved

// solve the test problems with dopri5 or radau5, from and to the binary files made by check.py
//   run <linear|vanderpol> <dopri5|radau5> <dir>

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <cudaode.cuh>
#include <radau5/method.cuh>
#include "problems.cuh"

using T = double;
using Events = cuda::ode::dopri5::FixedEvents<T>;

// read a binary file into managed memory
static T* load(const std::string & name, std::size_t n)
{
    T* p;
    cudaSafeCall(cudaMallocManaged(&p, n*sizeof(T)));
    std::ifstream in(name, std::ios::binary);
    in.read(reinterpret_cast<char*>(p), n*sizeof(T));
    if (!in) { std::fprintf(stderr, "can't read %s\n", name.c_str()); std::exit(1); }
    return p;
}

template <class Method, class Ode>
static void solve(Ode & ode, const std::string & dir, const T t1, const int neval, const T atol, const T rtol)
{
    using Solver = cuda::ode::dopri5::Solver<T, Ode, Events, Method>;
    auto N = ode.system_size;
    auto systems = ode.systems;
    Events events(0, t1);
    auto y0 = load(dir + "/y0.bin", N);
    auto teval = load(dir + "/teval.bin", neval);
    T* yeval;
    cudaSafeCall(cudaMallocManaged(&yeval, (std::size_t)systems*neval*N*sizeof(T)));

    auto start = std::chrono::steady_clock::now();
    Solver solver(ode, events, atol, rtol, systems, 0);
    solver.set_dense_output(neval, teval, yeval);
    solver.set_init_values(y0, true, systems, 0);
    solver.solve_ivp(true, systems, 0);
    auto stats = solver.statistics(systems);
    auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    std::ofstream out(dir + "/yeval.bin", std::ios::binary);
    out.write(reinterpret_cast<char*>(yeval), (std::size_t)systems*neval*N*sizeof(T));
    std::ofstream log(dir + "/stats.txt");
    for (auto & s : stats)
        log << s.accepted << " " << s.rejected << " " << s.stiff << " " << s.hmin << " " << s.hmax << "\n";
    std::printf("solved %d systems in %.2fs\n", systems, seconds);
}

template <class Method>
static void run(const std::string & problem, const std::string & dir)
{
    // the sizes: systems N L neval t1 atol rtol
    std::ifstream in(dir + "/sizes.txt");
    int systems, N, L, neval;
    T t1, atol, rtol;
    in >> systems >> N >> L >> neval >> t1 >> atol >> rtol;
    if (problem == "linear") {
        auto M = N - L;
        LinearOde<T> ode { N, 1, N, systems, L, M,
            load(dir + "/A.bin", (std::size_t)M*M), load(dir + "/B.bin", (std::size_t)L*M),
            load(dir + "/factor.bin", systems) };
        solve<Method>(ode, dir, t1, neval, atol, rtol);
    }
    else {
        VanDerPolOde<T> ode { 2, 1, 2, systems, load(dir + "/mu.bin", systems) };
        solve<Method>(ode, dir, t1, neval, atol, rtol);
    }
}

int main(int argc, char* argv[])
{
    if (argc != 4) { std::fprintf(stderr, "usage: run <linear|vanderpol> <dopri5|radau5> <dir>\n"); return 1; }
    std::string problem = argv[1], method = argv[2], dir = argv[3];
    if (method == "radau5")
        run<cuda::ode::radau5::Radau5<T>>(problem, dir);
    else
        run<cuda::ode::dopri5::Dopri5<T>>(problem, dir);
    return 0;
}

// end of file
