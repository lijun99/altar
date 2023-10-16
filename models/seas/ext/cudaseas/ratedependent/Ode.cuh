/*
 *  This is an example of an ode function required by the package
 *
 */

#ifndef __rd_ode_cuh__
#define __rd_ode_cuh__

// an example ode function
// please follow(copy) this example for naming conventions
template <class T>
struct __ALIGNED__ RateDependentODE {
    // required parameters, keep their names
    // @note each system is defined by #patches and each patch with #units
    // @note each patch is processed by one thread
    int patches; // number of patches per system
    int units;   // number of units per patch
    int system_size; // patches*units
    int systems; // total systems/samples to be processed

    // other custom parameters
    // all these parameters need to set inside this structure
    const T* alpha_h; // [systems * patches]
    const T* K_int; // [patches, 2, patches, 2]
    const T* K_ext; // [patches * 2]
    const T* v_p; // [patches * 2]
    T mu_over_2vs;
    T v_0;

    // define indexing functions

    // for K_inner_inner_onfault, 4D
    // K_int [patches, 2, patches, 2]
    __device__ __forceinline__ int i_Kii (int i0, int i1, int i2, int i3) {
        assert((i0 < patches) && (i1 < 2) && (i2 < patches) && (i3 < 2));
        return (i3) + (i2 * 2) + (i1 * 2 * patches) + (i0 * 2 * patches * 2);
    }

    // ode function called when solving a system with a thread block
    // y0, f [2(quantity: displacement, stress), 2(component: dip,strike), patches]
    // f[0, :, :] = v = v0 exp(y0[1, :, :])
    // f[1, :, :] = tau = [K_ext (v-vp) - K_int] / (mu_over_2vs * v - alpha_h)
    // TBD - since ODE independent of displacement, may use only stress/velocity
    // K_ext[component, patches]
    // v_p [patches]
    // K_int [patches, component, patches, component]
    // TBD - maybe better reshaped as [component, patches, component, patches] to use matrix-matrix product form
    // alpha_h [patches]

    __device__ __forceinline__  void dydt_block(const cg::thread_block& cta, const int system_id, const T t, const T* y0, T* f)
    {
        // update slip in both directions
        for (int patch_id = cta.thread_rank(); patch_id < patches; patch_id += cta.size()) {
            f[patch_id] = v_0 * exp(y0[patch_id + 2 * patches]);
            f[patch_id + patches] = v_0 * exp(y0[patch_id + 3 * patches]);
        }

        // wait for completion
        cta.sync();

        // update velocities in both directions
        for (int patch_id = cta.thread_rank(); patch_id < patches; patch_id += cta.size()) {
            // initialize with external influence
            f[patch_id + 2 * patches] = -K_ext[patch_id];
            f[patch_id + 3 * patches] = -K_ext[patch_id + patches];
            // tensor product with all other patches
            for (auto j = 0; j < patches; j++) {
                auto delv0 = f[j] - v_p[j];
                auto delv1 = f[j + patches] - v_p[j + patches];
                f[patch_id + 2 * patches] += K_int[i_Kii(patch_id, 0, j, 0)] * delv0 + K_int[i_Kii(patch_id, 0, j, 1)] * delv1;
                f[patch_id + 3 * patches] += K_int[i_Kii(patch_id, 1, j, 0)] * delv0 + K_int[i_Kii(patch_id, 1, j, 1)] * delv1;
            }
            // rescaling due to radiation damping
            f[patch_id + 2 * patches] /= mu_over_2vs * f[patch_id] + alpha_h[system_id * patches + patch_id];
            f[patch_id + 3 * patches] /= mu_over_2vs * f[patch_id + patches] + alpha_h[system_id * patches + patch_id];
        }

    };

    // debugging descriptor
    void describe() {

        cudaDeviceSynchronize();
        printf("RateDependentODE\n");
        printf("patches = %i, units = %i, system_size = %i, systems = %i\n",
               patches, units, system_size, systems);
        printf("mu_over_2vs = %g, v_0 = %g\n", mu_over_2vs, v_0);
        printf("alpha_h = %g ... %g\n", alpha_h[0], alpha_h[systems * patches - 1]);
        printf("K_int = %g ... %g\n", K_int[0], K_int[patches * patches * 4 - 1]);
        printf("K_ext = %g ... %g\n", K_ext[0], K_ext[2 * patches - 1]);
        printf("v_p = %g ... %g\n", v_p[0], v_p[2 * patches - 1]);

    }

    // constructor
    RateDependentODE(const int p, const int u, const int sys, const T* alpha_h_vec_, const T mu_over_2vs_, const T v_0_,
              const T* K_inner_inner_onfault_, const T* K_inner_asperities_v_plate_, const T* v_plate_ddcs_proj_eff_inner_)
        : patches(p), units(u), systems(sys), system_size(p*u), mu_over_2vs(mu_over_2vs_), v_0(v_0_),
          alpha_h(alpha_h_vec_), K_int(K_inner_inner_onfault_), K_ext(K_inner_asperities_v_plate_), v_p(v_plate_ddcs_proj_eff_inner_)
        {
            // describe();
        }
};

#endif //__rd_ode_cuh__
