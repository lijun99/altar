// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2022 - 2023 california institute of technology
// all rights reserved

// an example with iteratively solving ode for many cycles
// compile with, e.g., nvcc -arch=native example.cu -I../../

#include <cstdio>
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <vector>
#include <stdexcept>
#include <numeric>

// json library from https://github.com/nlohmann/json
#include "json.hpp"
using json = nlohmann::json;

// dopri5 solver
#include "cudaode.cuh"

// my ode and event definitions
#include "ode.cuh"
#include "event.cuh"

// binary file reader
template<class T> std::vector<T> read_bin_file(std::string filepath, int num) {
    // variables
    std::ifstream ifs(filepath, std::ios::in | std::ios::binary);
    std::vector<T> output;
    output.resize(num);
    // check for reading errors
    if (not ifs.read(reinterpret_cast<char*>(output.data()), sizeof(T) * num))
        throw std::runtime_error("Could not read file: '" + filepath + "'");
    // done
    ifs.close();
    return output;
}

// multiply all elements of a vector
int get_vector_size(std::vector<int> v) {
    return std::accumulate(v.begin(), v.end(), 1, std::multiplies<>());
}

// check for equality and if not fail verbosely
template<class T> void assert_number(std::string label, T left, T right) {
    if (left != right) throw std::runtime_error(label + ": " + std::to_string(left) + " != " + std::to_string(right));
}

// check for equality and if not fail verbosely
void assert_string(std::string label, std::string left, std::string right) {
    if (left != right) throw std::runtime_error(label + ": " + left + " != " + right);
}

// print vector info
void vector_stats(std::string label, std::vector<double> v) {
    printf("%s: size=%i, min=%g, max=%g\n",
           label.c_str(), v.size(), *std::min_element(v.begin(), v.end()), *std::max_element(v.begin(), v.end()));
    return;
}
void vector_stats(std::string label, std::vector<int> v) {
    printf("%s: size=%i, min=%i, max=%i\n",
           label.c_str(), v.size(), *std::min_element(v.begin(), v.end()), *std::max_element(v.begin(), v.end()));
    return;
}

int main()
{
    // some type renames
    using T = double;
    using EventType = SEASEvents<T>;
    using OdeType = RateDependentODE<T>;
    using SolverType = cuda::ode::dopri5::SpinupSolver<T, OdeType, EventType>;
    // using SolverType = cuda::ode::dopri5::Solver<T, OdeType, EventType>; // single cycle
    const bool load_second_sim = true;

    // look for runfiles.json in current folder
    std::string rootdir = "./";
    std::ifstream infofile(rootdir + "runfiles.json");
    if (infofile.fail()){
        std::cerr << "'" + rootdir + "runfiles.json' does not exist.";
        exit(EXIT_FAILURE);
    }

    // read into json object
    json info = json::parse(infofile);

    // t_eval_joint_sec (num_t_eval, )
    assert_string("t_eval_joint_sec.dtype", info.at("t_eval_joint_sec").at("dtype"), "float64");
    auto shape = info.at("t_eval_joint_sec").at("shape").get<std::vector<int>>();
    assert_number("t_eval_joint_sec.shape.size", shape.size(), (size_t) 1);
    auto num_t_eval = shape[0];
    std::vector<double> t_eval_joint_sec(read_bin_file<double>(rootdir + "t_eval_joint_sec.bin", num_t_eval));
    vector_stats("t_eval_joint_sec", t_eval_joint_sec);

    // ix_break_joint (num_ix_break, )
    assert_string("ix_break_joint.dtype", info.at("ix_break_joint").at("dtype"), "int32");
    shape = info.at("ix_break_joint").at("shape").get<std::vector<int>>();
    assert_number("ix_break_joint.shape.size", shape.size(), (size_t) 1);
    int num_ix_break = shape[0];
    std::vector<int> ix_break_joint(read_bin_file<int>(rootdir + "ix_break_joint.bin", num_ix_break));
    vector_stats("ix_break_joint", ix_break_joint);
    std::copy(ix_break_joint.begin(),
              ix_break_joint.end(),
              std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;

    // ix_eq_joint (num_ix_eq, )
    assert_string("ix_eq_joint.dtype", info.at("ix_eq_joint").at("dtype"), "int32");
    shape = info.at("ix_eq_joint").at("shape").get<std::vector<int>>();
    assert_number("ix_eq_joint.shape.size", shape.size(), (size_t) 1);
    auto num_ix_eq = shape[0];
    std::vector<int> ix_eq_joint(read_bin_file<int>(rootdir + "ix_eq_joint.bin", num_ix_eq));
    vector_stats("ix_eq_joint", ix_eq_joint);
    std::copy(ix_eq_joint.begin(),
              ix_eq_joint.end(),
              std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;

    // K_inner_inner_onfault (num_inner_patches, 2, num_inner_patches, 2)
    assert_string("K_inner_inner_onfault.dtype", info.at("K_inner_inner_onfault").at("dtype"), "float64");
    shape = info.at("K_inner_inner_onfault").at("shape").get<std::vector<int>>();
    assert_number("K_inner_inner_onfault.shape.size", shape.size(), (size_t) 4);
    int num_inner_patches = shape[0];
    assert_number("K_inner_inner_onfault.shape[0,2]", shape[0], shape[2]);
    assert_number("K_inner_inner_onfault.shape[1,3]", shape[1], shape[3]);
    assert_number("K_inner_inner_onfault.shape[1&3]", (int) shape[1], 2);
    std::vector<double> K_inner_inner_onfault(read_bin_file<double>(rootdir + "K_inner_inner_onfault.bin", get_vector_size(shape)));
    vector_stats("K_inner_inner_onfault", K_inner_inner_onfault);

    // K_inner_asperities_v_plate (num_inner_patches, 2)
    assert_string("K_inner_asperities_v_plate.dtype", info.at("K_inner_asperities_v_plate").at("dtype"), "float64");
    shape = info.at("K_inner_asperities_v_plate").at("shape").get<std::vector<int>>();
    assert_number("K_inner_asperities_v_plate.shape.size", shape.size(), (size_t) 2);
    assert_number("K_inner_asperities_v_plate.shape[0]", (int) shape[0], num_inner_patches);
    assert_number("K_inner_asperities_v_plate.shape[1]", (int) shape[1], 2);
    std::vector<double> K_inner_asperities_v_plate(read_bin_file<double>(rootdir + "K_inner_asperities_v_plate.bin", get_vector_size(shape)));
    vector_stats("K_inner_asperities_v_plate", K_inner_asperities_v_plate);

    // v_plate_ddcs_proj_eff_inner (num_inner_patches, 2)
    assert_string("v_plate_ddcs_proj_eff_inner.dtype", info.at("v_plate_ddcs_proj_eff_inner").at("dtype"), "float64");
    shape = info.at("v_plate_ddcs_proj_eff_inner").at("shape").get<std::vector<int>>();
    assert_number("v_plate_ddcs_proj_eff_inner.shape.size", shape.size(), (size_t) 2);
    assert_number("v_plate_ddcs_proj_eff_inner.shape[0]", (int) shape[0], num_inner_patches);
    assert_number("v_plate_ddcs_proj_eff_inner.shape[1]", (int) shape[1], 2);
    std::vector<double> v_plate_ddcs_proj_eff_inner(read_bin_file<double>(rootdir + "v_plate_ddcs_proj_eff_inner.bin", get_vector_size(shape)));
    vector_stats("v_plate_ddcs_proj_eff_inner", v_plate_ddcs_proj_eff_inner);

    // v_init (num_inner_patches, 2)
    assert_string("v_init.dtype", info.at("v_init").at("dtype"), "float64");
    shape = info.at("v_init").at("shape").get<std::vector<int>>();
    assert_number("v_init.shape.size", shape.size(), (size_t) 2);
    assert_number("v_init.shape[0]", (int) shape[0], num_inner_patches);
    assert_number("v_init.shape[1]", (int) shape[1], 2);
    std::vector<double> v_init(read_bin_file<double>(rootdir + "v_init.bin", get_vector_size(shape)));
    vector_stats("v_init", v_init);

    // delta_tau_bounded (num_eq, num_inner_patches, 2)
    assert_string("delta_tau_bounded.dtype", info.at("delta_tau_bounded").at("dtype"), "float64");
    shape = info.at("delta_tau_bounded").at("shape").get<std::vector<int>>();
    assert_number("delta_tau_bounded.shape.size", shape.size(), (size_t) 3);
    int num_eq = shape[0];
    assert_number("delta_tau_bounded.shape[1]", (int) shape[1], num_inner_patches);
    assert_number("delta_tau_bounded.shape[2]", (int) shape[2], 2);
    std::vector<double> delta_tau_bounded(read_bin_file<double>(rootdir + "delta_tau_bounded.bin", get_vector_size(shape)));
    vector_stats("delta_tau_bounded", delta_tau_bounded);

    // load second delta_tau_bounded
    if (load_second_sim) {
        std::vector<double> delta_tau_bounded2(read_bin_file<double>(rootdir + "delta_tau_bounded2.bin", get_vector_size(shape)));
        vector_stats("delta_tau_bounded2", delta_tau_bounded2);
        // append
        delta_tau_bounded.insert(delta_tau_bounded.end(), delta_tau_bounded2.begin(), delta_tau_bounded2.end());
    }

    // v_0
    assert_string("v_0.dtype", info.at("v_0").at("dtype"), "float64");
    shape = info.at("v_0").at("shape").get<std::vector<int>>();
    assert_number("v_0.shape.size", shape.size(), (size_t) 1);
    double v_0(read_bin_file<double>(rootdir + "v_0.bin", 1)[0]);
    printf("v_0: %g\n", v_0);

    // alpha_h_vec (num_inner_patches, )
    assert_string("alpha_h_vec.dtype", info.at("alpha_h_vec").at("dtype"), "float64");
    shape = info.at("alpha_h_vec").at("shape").get<std::vector<int>>();
    assert_number("alpha_h_vec.shape.size", shape.size(), (size_t) 1);
    assert_number("alpha_h_vec.shape[0]", (int) shape[0], num_inner_patches);
    std::vector<double> alpha_h_vec(read_bin_file<double>(rootdir + "alpha_h_vec.bin", num_inner_patches));
    vector_stats("alpha_h_vec", alpha_h_vec);

    // load second alpha_h_vec
    if (load_second_sim) {
        std::vector<double> alpha_h_vec2(read_bin_file<double>(rootdir + "alpha_h_vec2.bin", num_inner_patches));
        vector_stats("alpha_h_vec2", alpha_h_vec2);
        // append
        alpha_h_vec.insert(alpha_h_vec.end(), alpha_h_vec2.begin(), alpha_h_vec2.end());
    }

    // mu_over_2vs
    assert_string("mu_over_2vs.dtype", info.at("mu_over_2vs").at("dtype"), "float64");
    shape = info.at("mu_over_2vs").at("shape").get<std::vector<int>>();
    assert_number("mu_over_2vs.shape.size", shape.size(), (size_t) 1);
    double mu_over_2vs(read_bin_file<double>(rootdir + "mu_over_2vs.bin", 1)[0]);
    printf("mu_over_2vs: %g\n", mu_over_2vs);

    // index function for K_inner_asperities_v_plate, v_plate_ddcs_proj_eff_inner, v_init, 2D
    auto i_Kia_v = [num_inner_patches](int i0, int i1) {
        if ((i0 >= num_inner_patches) || (i1 >= 2))
            throw std::runtime_error("i_Kia_v: ["
                                     + std::to_string(i0) + ","
                                     + std::to_string(i1) + "] outside of shape ("
                                     + std::to_string(num_inner_patches) + ",2)");
        return (i1) + (i0 * 2);
    };

    // define ode system
    const int units = 4;
    const int systems_batch = 2;
    const int max_cycles = 20;
    int systems;
    if (load_second_sim) {
        systems = 2;
    }
    else {
        systems = 1;
    }

    // variable sizes
    int patches = num_inner_patches;
    int system_size = patches * units;

    // initialize ODE
    OdeType odefunc {patches, units, systems, alpha_h_vec.data(), mu_over_2vs, v_0, K_inner_inner_onfault.data(),
                     K_inner_asperities_v_plate.data(), v_plate_ddcs_proj_eff_inner.data()};

    // copy initial state in the form of first all slips, then all velocities
    T *y0;
    cudaMallocManaged(&y0, system_size * sizeof(T));
    for (auto i=0; i<patches; i++) {
        y0[i] = (T) 0;
        y0[i + patches] = (T) 0;
        y0[i + 2 * patches] = (T) log(v_init[i_Kia_v(i, 0)] / v_0);
        y0[i + 3 * patches] = (T) log(v_init[i_Kia_v(i, 1)] / v_0);
    }

    // extract evaluation times

    // // first cycle (without observation times)
    // int ix_break_start = 0;
    // int ix_break_stop = ix_break_joint[1];

    // last cycle (with observation times)
    // the dense teval is the last cycle of t_eval_joint_sec
    // bounded by the last two indices in ix_break_joint
    // need to find it and reset to zero at cycle start
    int ix_break_start = ix_break_joint[num_ix_break - 2];
    int ix_break_stop = ix_break_joint[num_ix_break - 1];

    // initialize teval
    int neval = ix_break_stop - ix_break_start + 1;
    T * teval;
    cudaMallocManaged(&teval, neval * sizeof(T));
    for (auto i=0; i<neval; i++)
        teval[i] = t_eval_joint_sec[ix_break_start + i] - t_eval_joint_sec[ix_break_start];

    // print info
    printf("ix_break: %i -> %i\n", ix_break_start, ix_break_stop);
    printf("t_eval_joint_sec: %g -> %g\n",
           t_eval_joint_sec[ix_break_start], t_eval_joint_sec[ix_break_stop]);
    printf("teval: size=%i, first=%g, last=%g\n",
           neval, teval[0], teval[neval-1]);

    // set myevents
    // the event times are defined in ix_eq_joint and are indices between
    // ix_break_start and ix_break_end
    // find first event index

    // // for first cycle
    // int ix_eq_start = 0;
    // int nevents = -1;
    // for (auto i=0; i<num_ix_eq; i++) {
    //     if (ix_eq_joint[i] >= ix_break_joint[1]) {
    //         nevents = i + 2;
    //         break;
    //     }
    // }
    // if (nevents == -1) throw std::runtime_error("!! nevents == -1");

    // for last cycle
    int ix_eq_start = -1;
    for (auto i=0; i<num_ix_eq; i++) {
        if (ix_eq_joint[i] >= ix_break_start) {
            ix_eq_start = i;
            break;
        }
    }
    int nevents = num_ix_eq - ix_eq_start + 2;
    if (ix_eq_start == -1) throw std::runtime_error("!! ix_eq_start == -1");

    // extract event times from t_eval_joint_sec
    // current format of nevents has to include the start and end time
    std::vector<double> tevents(nevents);
    tevents[0] = teval[0];
    tevents[nevents-1] = teval[neval - 1];
    for (auto i=0; i<nevents-2; i++)
        tevents[i+1] = t_eval_joint_sec[ix_eq_joint[ix_eq_start + i]] - t_eval_joint_sec[ix_break_start];

    // print info
    printf("ix_eq_start=%i\nnevents=%i\ntevents=[ ", ix_eq_start, nevents);
    std::copy(tevents.begin(), tevents.end(),
              std::ostream_iterator<double>(std::cout, " "));
    printf("]\n");

    // initialize events
    EventType events {nevents, tevents.data(), delta_tau_bounded.data(), alpha_h_vec.data(), systems, patches, units};

    // solver settings
    bool dense_out = true;
    bool use_y0_for_all = true;
    const T atol = 1e-8;
    const T rtol = 1e-6;

    // initialize dense output array
    T * yeval;
    cudaMallocManaged(&yeval, systems * neval * system_size * sizeof(T));

    // construct the ode solver
    SolverType solver{odefunc, events, atol, rtol, 1e-6, 1e-3, systems_batch};
    // set dense output
    solver.set_dense_output(neval, teval, yeval);

    // iteratively solve systems in batch
    int conv_i_start = (units / 2) * patches;
    int conv_i_stop = units * patches - 1;
    for (int system_offset=0; system_offset<systems; system_offset+=systems_batch)
    {
        // check how many systems are left
        auto systems_to_process = min(systems_batch, systems-system_offset);
         // set initial values
        solver.set_init_values(y0, use_y0_for_all, systems_to_process, system_offset);
        // call the solver
        printf("iterate systems %d %d %d ... \n", system_offset, systems_to_process, systems);
        solver.solve_ivp_cycles(dense_out, systems_to_process, system_offset,
                                conv_i_start, conv_i_stop, max_cycles);
        // solver.solve_ivp(dense_out, systems_to_process, system_offset); // single cycle

        cudaDeviceSynchronize();
    }

    // write yeval output file
    std::string ypath = "yeval.bin";
    std::ofstream yfile(ypath, std::ios::out | std::ios::binary);
    if (!yfile) {throw std::runtime_error("Could not write file: '" + ypath + "'");}
    yfile.write((char *) &yeval[0], systems * neval * system_size * sizeof(T));
    yfile.close();

    // write teval output file
    std::string tpath = "teval.bin";
    std::ofstream tfile(tpath, std::ios::out | std::ios::binary);
    if (!tfile) {throw std::runtime_error("Could not write file: '" + tpath + "'");}
    tfile.write((char *) &teval[0], neval * sizeof(T));
    tfile.close();

    // print output file info
    printf("Output file shape = %i (systems=%i, neval=%i, system_size=%i)\n",
           systems * neval * system_size, systems, neval, system_size);

}
