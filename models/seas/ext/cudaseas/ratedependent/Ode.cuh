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
    T* alpha_h; // [systems * patches]
    T* K_int; // [patches, 2, patches, 2]
    T* K_ext; // [patches * 2]
    T* v_p; // [patches * 2]
    T mu_over_2vs;
    T v_0;

    // define indexing functions

    // for K_inner_inner_onfault, 4D
    __device__ __forceinline__ int i_Kii (int i0, int i1, int i2, int i3) {
        assert((i0 < patches) && (i1 < 2) && (i2 < patches) && (i3 < 2));
        return (i3) + (i2 * 2) + (i1 * 2 * patches) + (i0 * 2 * patches * 2);
    }

    // for K_inner_asperities_v_plate, v_plate_ddcs_proj_eff_inner, v_init, 2D
    int i_Kia_v (int i0, int i1) {
        assert((i0 < patches) && (i1 < 2));
        return (i1) + (i0 * 2);
    }

    // ode function called when solving a system with a thread block
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

    // constructor
    RateDependentODE(const int p, const int u, const int sys, const T* alpha_h_vec_, const T mu_over_2vs_, const T v_0_,
              const T* K_inner_inner_onfault_, const T* K_inner_asperities_v_plate_, const T* v_plate_ddcs_proj_eff_inner_)
        : patches(p), units(u), systems(sys), system_size(p*u), mu_over_2vs(mu_over_2vs_), v_0(v_0_)
    {
        // set up alpha_h_vec
        cudaMallocManaged(&alpha_h, systems * patches * sizeof(T));
        cudaMemcpy(alpha_h, alpha_h_vec_, systems * patches * sizeof(T), cudaMemcpyDefault);

        // set up internal stress kernel - this is a 4D tensor that we'll have to
        // correctly index within dydt
        auto num_p_p_4 = patches * patches * 4;
        cudaMallocManaged(&K_int, num_p_p_4 * sizeof(T));
        cudaMemcpy(K_int, K_inner_inner_onfault_, num_p_p_4 * sizeof(T), cudaMemcpyDefault);

        // for K_ext and v_p it makes sense to already reshape them to match the format of f and y,
        // i.e. first all the velocities for unit 1, then all the valocities for unit 2

        // set up external stress kernel and plate velocities
        cudaMallocManaged(&K_ext, patches * 2 * sizeof(T));
        cudaMallocManaged(&v_p, patches * 2 * sizeof(T));
        for (auto i = 0; i < patches; i++) {
            auto i0 = i_Kia_v(i, 0);
            auto i1 = i_Kia_v(i, 1);
            K_ext[i] = K_inner_asperities_v_plate_[i0];
            K_ext[i + patches] = K_inner_asperities_v_plate_[i1];
            v_p[i] = v_plate_ddcs_proj_eff_inner_[i0];
            v_p[i + patches] = v_plate_ddcs_proj_eff_inner_[i1];
        }
    };
};

#endif //__rd_ode_cuh__
