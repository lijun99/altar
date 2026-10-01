// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2026 california institute of technology
// all rights reserved

// test problems for the radau5 method

#ifndef radau5_tests_problems_cuh
#define radau5_tests_problems_cuh

#include <dopri5/external.h>

// s' = B z, z' = f A z: stiff and linear, with leading components the derivatives don't depend on
template <class T>
struct LinearOde {
    int patches; // N, one unit per patch
    int units;
    int system_size;
    int systems;
    int L; // inert components
    int M; // the others
    const T* A; // [M, M]
    const T* B; // [L, M]
    const T* factor; // [systems]

    __host__ __device__ int inert_size() const { return L; }

    __device__ void dydt_block(const cg::thread_block& cta, const int system_id, const T t, const T* y, T* f)
    {
        auto z = y + L;
        auto s = factor[system_id];
        for (auto i = static_cast<int>(cta.thread_rank()); i < L + M; i += cta.size()) {
            auto acc = static_cast<T>(0);
            if (i < L)
                for (auto k = 0; k < M; k++) acc += B[i*M + k]*z[k];
            else
                for (auto k = 0; k < M; k++) acc += s*A[(i - L)*M + k]*z[k];
            f[i] = acc;
        }
    }
};

// van der Pol: y0' = y1, y1' = mu (1 - y0^2) y1 - y0
template <class T>
struct VanDerPolOde {
    int patches;
    int units;
    int system_size;
    int systems;
    const T* mu; // [systems]

    __device__ void dydt_block(const cg::thread_block& cta, const int system_id, const T t, const T* y, T* f)
    {
        if (cta.thread_rank() == 0) {
            f[0] = y[1];
            f[1] = mu[system_id]*(1 - y[0]*y[0])*y[1] - y[0];
        }
    }
};

#endif
// end of file
