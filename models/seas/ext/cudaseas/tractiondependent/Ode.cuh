/*
 *  Traction-dependent ODE function
 *  Same as RateDependentODE but with an additional constant rho
 */

#include <cmath>
// #include <assert.h>
#include "wright_omega.cuh"

#ifndef __td_ode_cuh__
#define __td_ode_cuh__

#define i_Kii(i0, i1, i2, i3, patches)  \
    ((i3) + (i2 * 2) + (i1 * 2 * patches) + (i0 * 2 * patches * 2))



// traction-dependent ode function
// identical to RateDependentODE with the addition of a constant rho
template <class T>
struct __ALIGNED__ TractionDependentODE {
    // required parameters, keep their names
    // @note each system is defined by #patches and each patch with #units
    // @note each patch is processed by one thread
    int patches; // number of patches per system
    int units;   // number of units per patch
    int system_size; // patches*units

    // the leading components the derivatives do not depend on, the slip, for implicit methods
    __host__ __device__ int inert_size() const { return 2*patches; }
    int systems; // total systems/samples to be processed

    // other custom parameters
    // all these parameters need to set inside this structure
    const T* alpha_h; // [systems * patches]
    const T* K_int; // [patches, 2, patches, 2]
    const T* K_ext; // [patches * 2]
    const T* v_p; // [patches * 2]
    const T mu_over_2vs;
    const T v_0;
    const T rho; // <--- NEW: dimensionless fricitonal parameter

    // ode function called when solving a system with a thread block
    //
    __device__ __forceinline__  void dydt_block(const cg::thread_block& cta, const int system_id, const T t, const T* y0, T* f)
    {
        // each thread processes one patch (or more if patches>threads)
        for (int patch_id = cta.thread_rank(); patch_id < patches; patch_id += cta.size()) {

            // put the magnitude of the traction into the first component of the output array
            // (temporarily)
            f[patch_id + 2 * patches] = sqrt(pow(y0[patch_id + 2 * patches], 2) +
                                             pow(y0[patch_id + 3 * patches], 2));

            // convert tau magnitude to velocity magnitude, put it in the second traction component
            // (temporarily)
            auto minusc = mu_over_2vs / alpha_h[system_id * patches + patch_id];
            auto zprime = (
                f[patch_id + 2 * patches] / alpha_h[system_id * patches + patch_id]
                - rho + log(v_0 * minusc)
            );
            f[patch_id + 3 * patches] = wright_omega(zprime) / minusc;

            // now we have both the traction and the velocity magnitudes,
            // compute the velocity vector and put it into its final place
            auto v_tau_ratio = f[patch_id + 3 * patches] / f[patch_id + 2 * patches];
            f[patch_id] = y0[patch_id + 2 * patches] * v_tau_ratio;
            f[patch_id + patches] = y0[patch_id + 3 * patches] * v_tau_ratio;
        }

        // wait for completion
        cta.sync();

        // get the elastic traction derivative from the boundary elements integral
        for (int patch_id = cta.thread_rank(); patch_id < patches; patch_id += cta.size()) {
            // initialize with external influence
            f[patch_id + 2 * patches] = -K_ext[patch_id];
            f[patch_id + 3 * patches] = -K_ext[patch_id + patches];
            // tensor product with all other patches
            for (auto j = 0; j < patches; j++) {
                auto delv0 = f[j] - v_p[j];
                auto delv1 = f[j + patches] - v_p[j + patches];
                f[patch_id + 2 * patches] += K_int[i_Kii(patch_id, 0, j, 0, patches)] * delv0 + K_int[i_Kii(patch_id, 0, j, 1, patches)] * delv1;
                f[patch_id + 3 * patches] += K_int[i_Kii(patch_id, 1, j, 0, patches)] * delv0 + K_int[i_Kii(patch_id, 1, j, 1, patches)] * delv1;
            }
        }
        //cta.sync();

    };

    // the Jacobian of my derivatives f = dydt(t, y) with respect to the traction, for implicit
    // methods: J[i*M + c] = df_i / dy_(2P + c), for all 4P components i and the M = 2P tractions c,
    // with {work} holding the 2x2 blocks dv/dtau of the patches, 4P reals
    __device__ __forceinline__ void jacobian_block(const cg::thread_block& cta, const int system_id, const T t,
                                                   const T* y, const T* f, T* J, T* work)
    {
        auto M = 2 * patches;
        // the blocks dv_a/dtau_b of each patch, from the slip rates in f
        for (int p = cta.thread_rank(); p < patches; p += cta.size()) {
            auto t1 = y[p + 2 * patches];
            auto t2 = y[p + 3 * patches];
            auto m = sqrt(t1 * t1 + t2 * t2);
            auto a = alpha_h[system_id * patches + p];
            // the slip rate magnitude, and omega = v mu/(2 vs alpha_h), the value of the wright omega
            auto v = sqrt(f[p] * f[p] + f[p + patches] * f[p + patches]);
            auto omega = v * mu_over_2vs / a;
            // dv/dm, and v/m
            auto dv = v / ((1 + omega) * a);
            auto r = v / m;
            auto q = (dv - r) / (m * m);
            work[4 * p + 0] = r + t1 * t1 * q;
            work[4 * p + 1] = t1 * t2 * q;
            work[4 * p + 2] = t2 * t1 * q;
            work[4 * p + 3] = r + t2 * t2 * q;
        }
        cta.sync();
        // the slip rows are the blocks themselves, the traction rows the elastic interactions of them
        for (int idx = cta.thread_rank(); idx < 2 * M * M; idx += cta.size()) {
            auto i = idx / M;
            auto c = idx % M;
            auto b = c / patches;
            auto j = c % patches;
            T value;
            if (i < M) {
                auto component = i / patches;
                value = (i % patches == j) ? work[4 * j + 2 * component + b] : static_cast<T>(0);
            }
            else {
                auto component = (i - M) / patches;
                auto patch = (i - M) % patches;
                value = K_int[i_Kii(patch, component, j, 0, patches)] * work[4 * j + b]
                      + K_int[i_Kii(patch, component, j, 1, patches)] * work[4 * j + 2 + b];
            }
            J[(size_t)i * M + c] = value;
        }
        cta.sync();
    }

    // debugging descriptor
    void describe() {

        cudaDeviceSynchronize();
        printf("TractionDependentODE\n");
        printf("patches = %i, units = %i, system_size = %i, systems = %i\n",
               patches, units, system_size, systems);
        printf("mu_over_2vs = %g, v_0 = %g, rho = %g\n", mu_over_2vs, v_0, rho);
        printf("alpha_h = %g ... %g\n", alpha_h[0], alpha_h[systems * patches - 1]);
        printf("K_int = %g ... %g\n", K_int[0], K_int[patches * patches * 4 - 1]);
        printf("K_ext = %g ... %g\n", K_ext[0], K_ext[2 * patches - 1]);
        printf("v_p = %g ... %g\n", v_p[0], v_p[2 * patches - 1]);

    }

    // constructor
    TractionDependentODE(const int p, const int u, const int sys,
              const T* alpha_h_vec_, const T mu_over_2vs_, const T v_0_, const T rho_,
              const T* K_inner_inner_onfault_, const T* K_inner_asperities_v_plate_,
              const T* v_plate_ddcs_proj_eff_inner_)
        : patches(p), units(u), systems(sys), system_size(p*u),
          mu_over_2vs(mu_over_2vs_), v_0(v_0_), rho(rho_),
          alpha_h(alpha_h_vec_), K_int(K_inner_inner_onfault_),
          K_ext(K_inner_asperities_v_plate_), v_p(v_plate_ddcs_proj_eff_inner_)
        {
            // describe();
        }

    // destructor
    ~TractionDependentODE()
    {
        // nothing to do
    }
};

#endif //__td_ode_cuh__
