// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2022 - 2023 california institute of technology
// all rights reserved
//

/*** This file includes routines to perform d_pred = G * M  in the forward model
 * M are the slip rate on patches, normally arranged as [samples (or systems), t_steps, patches*slip_components]
 * G are the Green's functions, arranged as [patches*components, stations*disp_components]
 * d_pred are the (predicted) surface displacements, arranged as [samples, t_steps, stations*disp_components]
 * d_obs are the (observed) surface displacements, arranged as [t_steps, stations*disp_components]
 * IF CONSIDERING Cd or Cp, we only consider Cd between stations*disp_components at a GIVEN t_step
 *    i.e., correlations between different times are neglected.
 * In this case,
 *  - Cd [t_steps, stations*disp_components, stations*disp_components]
 *  - if we choose to merge Cd into G and d_obs, i.e.,
 *        $Cd^{-1} = \chi \chi^T$, ${\tilde G} = G \chi$, ${\tilde d}_{obs} = d_{obs} \chi$.
 *   since Cd ($\chi$) is t_step dependent,
 *  - G [t_steps, patches*components, stations*disp_components]
 * N.B. extra t_steps dimension might lead to a huge size of Cd, G to be handled by GPU.
 *      May need to process t_step one by one.
***/

// code guard
#if !defined(altar_models_seas_cuda_displacement_steps_cuh)
#define altar_models_seas_cuda_displacement_steps_cuh

#include "cublas_wrapper.h" // C++ template wrapper for cublas routines

namespace altar::models::seas::cuda::ratedependent {

// Routine to copy velocity from sim_state to state_init for next run
// state_init [systems, 4*patches]
// sim_state [systems, n_obs, 4*patches], only the first num_forward_batch systems have been updated
// when copy to state_init, we only copy from systems which have been updated, in cyclic
template <typename T>
__global__ void copy_velocity_kernel(T* state_init,
    const T* sim_state,
    const int patches,
    const int num_t_obs,
    const int num_forward_batch)
{
    // thread blocks along x - systems*slip_size
    auto thread_id = threadIdx.x; // patches
    auto block_id = blockIdx.x; // system index


    auto sim_state_index = (block_id % num_forward_batch) * num_t_obs * 4 * patches;
    auto state_init_index = block_id * 4 * patches;

    // iterate over patches if total #patches > #threads per block
    for(auto patch = thread_id; patch < patches; patch += blockDim.x)
    {
        // get the slip rate from stress rate
        state_init[state_init_index + 2*patches + patch] = sim_state[sim_state_index + 2*patches + patch];
        state_init[state_init_index + 3*patches + patch] = sim_state[sim_state_index + 3*patches + patch];
    }
    // all done
}

// how to call the gpu kernel to copy slip
template <typename T>
void copy_velocity(T* state_init,
    const T* sim_state,
    const int patches,
    const int num_t_obs,
    const int num_forward_batch,
    const int cuda_batch_size,
    const int threads)
{
    // get the number of blocks
    auto blocks = cuda_batch_size;

    // call the kernel
    copy_velocity_kernel<T><<<blocks, threads>>>(
        state_init, sim_state, patches, num_t_obs, num_forward_batch);
    cudaCheckError("copy_velocity_kernel error");
    // all done
}


// Routine to copy slip from yeval/sim_state
// yeval/sim_state from ODE solvers are arranged as [systems, t_steps, 2(slip/stress), 2(slip_components), patches]
// we need to construct slip_history [systems, t_steps, 2(slip_components), patches]
//                 = sim_state[systems, t_steps, 1, 2(slip_components), patches]
// parallel scheme, use blocks = systems*t_steps, threads_per_block ~ patches
// each thread process two components
template <typename T>
__global__ void copy_slip_kernel(T* slip_history,
    const T* sim_state,
    const int patches,
    const T v_0)
{
    // thread blocks along x - systems*slip_size
    auto thread_id = threadIdx.x; // patches
    auto block_id = blockIdx.x; // systems*t_steps

    auto sim_state_index = block_id * 4 * patches;
    auto slip_rate_index = block_id * 2 * patches;

    // iterate over patches if total #patches > #threads per block
    for(auto patch = thread_id; patch < patches; patch += blockDim.x)
    {
        // get the slip rate from stress rate
        slip_history[slip_rate_index + patch] = sim_state[sim_state_index + patch];
        slip_history[slip_rate_index + patch + patches] = sim_state[sim_state_index + patch + patches];
    }
    // all done
}

// how to call the gpu kernel to copy slip
template <typename T>
void copy_slip(T* slip_history,
    const T* sim_state,
    const int systems,
    const int t_steps,
    const int patches,
    const T v_0,
    const int threads)
{
    // get the number of blocks
    auto blocks = systems*t_steps;

    // total threads = blocks * threads

    // call the kernel
    copy_slip_kernel<T><<<blocks, threads>>>(
        slip_history, sim_state, patches, v_0);
    cudaCheckError("copy_slip_kernel error");
    // all done
}

// Routine to convert slip rate in yeval/sim_state
// parallel scheme, use blocks = systems*t_steps, threads_per_block ~ patches
// each thread process two components
template <typename T>
__global__ void convert_slip_rate_kernel(
    T* sim_state,
    const int patches,
    const T v_0)
{
    // thread blocks along x - systems*slip_size
    auto thread_id = threadIdx.x; // patches
    auto block_id = blockIdx.x; // systems*t_steps

    auto sim_state_index = block_id * 4 * patches + 2 * patches;

    // iterate over patches if total #patches > #threads per block
    for(auto patch = thread_id; patch < patches; patch += blockDim.x)
    {
        // get the slip rate from stress rate
        sim_state[sim_state_index + patch] = v_0 * exp(sim_state[sim_state_index + patch]);
        sim_state[sim_state_index + patch + patches] = v_0 * exp(sim_state[sim_state_index + patch + patches]);
    }
    // all done
}

// how to call the gpu kernel to extract slip rate
template <typename T>
void convert_slip_rate(
    T* sim_state,
    const int systems,
    const int t_steps,
    const int patches,
    const T v_0,
    const int threads)
{
    // get the number of blocks
    auto blocks = systems*t_steps;

    // total threads = blocks * threads

    // call the kernel
    convert_slip_rate_kernel<T><<<blocks, threads>>>(sim_state, patches, v_0);
    cudaCheckError("convert_slip_rate_kernel error");
    // all done
}

// Implementation 1 - assume Cd is a constant (self-correlated, or diagonal) for all observed data.
// Gf is therefore the same for all samples and t_steps
// We compute d_pred [samples, t_steps, stations*displacement_components]
//               = slip_rate [samples, t_steps, patches*slip_components] * Gf[patches*slip_components, stations*displacement_components]
// This is achieved by batchedGemm, batch = samples/systems
// and for each batch/sample/system, perform a matrix-matrix product (gemm)
//     d_pred[t_steps, stations*displacement_components]
//         = slip_rate [t_steps, patches*slip_components] * Gf[patches*slip_components, stations*displacement_components]

template <typename T>
void compute_displacement_impl1(
    T* predictions, // [samples, t_steps, displacement_size], output d_pred,  displacement_size = stations * displacement_components
    const T* sim_state, // [samples, t_steps, patches*4]
    const T* gf, // [slip_size, displacement_size], input Green's function
    const int samples, // samples or systems
    const int t_steps, // time steps
    const int patches, //
    const int displacement_size, // stations*displacement_components
    const T v_0, // v_0
    const T gemm_alpha, // d = alpha G* M + beta d, alpha may be cd^{-1}
    const T gemm_beta,  // 0 or -alpha (if d_obs is copied to d to compute residue)
    const int threads
    )
{

    // printf("predictions needs to have shape (%i, %i, %i) = %i\n",
    //        samples, t_steps, displacement_size, samples * t_steps * displacement_size);
    // printf("sim_state needs to have shape (%i, %i, %i*4) = %i\n",
    //        samples, t_steps, patches, samples * t_steps * patches * 4);
    // copy slip history into [samples, t_steps, 2*patches]
    // auto system_size = patches * 4; // sim_state size per system per t_step
    auto slip_size = patches * 2; // slip rate size per system per t_step
    T* slip_history;
    std::size_t allocate_size = samples*t_steps*slip_size*sizeof(T);
    cudaSafeCall(cudaMallocManaged(&slip_history, allocate_size));
    // printf("gf needs to have shape (%i, %i) = %i\n",
    //        slip_size, displacement_size, slip_size * displacement_size);

    // copy slip rate from sim_state
    copy_slip<T>(slip_history, sim_state, samples, t_steps, patches, v_0, threads);
    // printf("copy slip done (%i, %i) = %i\n",
    //        slip_size, displacement_size, slip_size * displacement_size);

    // create a cublas handle
    cublasHandle_t handle;
    cublasSafeCall(cublasCreate(&handle));

    // Call Gemm Batched C = alpha A B + beta C
    // the batch size = samples or systems
    // For each system
    // slip [t_steps, slip_size] * G [slip_size, stations] = displacement [t_steps, displacement_size]
    // cublas use Column-major, which is equivalent to
    //  G [displacement_size, slip_size] * slip[slip_size, t_steps] = displacement [displacement_size, t_steps]
    cublasSafeCall(cuBLASGemmStridedBatched<T>(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        displacement_size, t_steps, slip_size,
        &gemm_alpha,
        gf, displacement_size, 0, // A, lda, strideA (0=same gf from all samples)
        slip_history, slip_size, slip_size*t_steps, // B, ldc, strideB
        &gemm_beta,
        predictions, displacement_size, displacement_size*t_steps,
        samples));

    // free handle
    cublasSafeCall(cublasDestroy(handle));

    // free slip_history
    cudaSafeCall(cudaFree(slip_history));

    // all done, return predictions as d_pred or d_pred-d_obs
}

// Implementation 2 - assume Cd matrix is different for different t_step: Cd[t_steps, displacement_size, displacement_size]
//   assume GPU memory size is large enough to hold Gf[t_steps, slip_size, displacement_size] and d_obs[t_steps, displacement_size]
//   it is recommended to merge Cd into Gf and d_obs, that's when Gf acquires t_step dependence
// We compute d_pred [samples, t_steps, displacement_size]
//               = slip_rate [samples, t_steps, slip_size] * Gf[t_steps, slip_size, displacement_size]
// We can still use batched gemm with batches = t_steps, or
//       d_pred' [t_steps, samples, displacement_size]
//               = slip_rate' [t_steps, samples, slip_size] * Gf[t_steps, slip_size, displacement_size]]
// This requires two additional transpose operations on slip_rate (before) and d_pred (after)
// NOTE: cutlass may have routines for tensor multiplications without transpose --- CHECK later

// Routine to extract slip rate from yeval/sim_state
// This is the same as slip_rate_kernel, but to transpose, or switch the indices of samples and t_steps
//
// WARNING still contains wrong case that the Green's functions is multiplying the velocities
// (rather than the cumulative displacement)
template <typename T>
__global__ void slip_rate_transpose_kernel(
    T* slip_rate,  // [t_steps, systems, 2*patches]
    const T* sim_state, // [systems, t_steps, 4*patches]
    const int systems,
    const int t_steps,
    const int patches,
    const T v_0)
{
    // thread blocks along x - systems*slip_size
    auto thread_id = threadIdx.x; // patches
    auto system_id = blockIdx.x; // systems
    auto tstep_id = blockIdx.y; // t_steps

    // note the change of indices of system_id and tstep_id
    auto sim_state_index = (system_id*t_steps + tstep_id) * 4 * patches + 2 * patches;
    auto slip_rate_index = (tstep_id*systems + system_id) * 2 * patches;

    // iterate over patches if total #patches > #threads per block
    for(auto patch = thread_id; patch < patches; patch += blockDim.x)
    {
        // get the slip rate from stress rate
        slip_rate[slip_rate_index + patch] = v_0 * exp(sim_state[sim_state_index + patch]);
        slip_rate[slip_rate_index + patch + patches] = v_0 * exp(sim_state[sim_state_index + patch + patches]);
    }
    // all done
}

// how to call the gpu kernel to extract slip rate
template <typename T>
void extract_slip_rate_transpose(T* slip_rate,
    const T* sim_state,
    const int systems,
    const int t_steps,
    const int patches,
    const T v_0,
    const int threads)
{
    // get the number of blocks
    dim3 threads_per_block(threads, 1); // #threads along x, y
    dim3 blocks(systems, t_steps);

    // call the kernel
    slip_rate_transpose_kernel<T><<<blocks, threads>>>(slip_rate, sim_state,
        systems, t_steps, patches, v_0);
    cudaCheckError("slip_rate_transpose_kernel error");
    // all done
}

// cuda kernel to transpose d[d0, d1, d2] to d'[d1, d0, d2]
template <typename T>
__global__ void tensor3d_transpose_102_kernel (T* d_out, const T* d_in,
    const int d0, const int d1, const int d2)
{
    // get index at each dimension [id0, id1, id2]
    int id2s = threadIdx.x + blockDim.x * blockIdx.x;
    int id1 = threadIdx.y + blockDim.y * blockIdx.y;
    int id0 = threadIdx.z + blockDim.z * blockIdx.z;

    if(id0 < d0 && id1 < d1)
    {
        for(auto id2 = id2s; id2 < d2; id2 += blockDim.x)
            d_out[(id1 * d0 + id0) * d2 + id2] = d_in[(id0 * d1 + id1) * d2 + id2];
    }
    // all done
}

// transpose d_pred from [t_steps, samples, displacement_size] to
//  samples, t_steps, displacement_size]
template <typename T>
void transpose_displacement(T* prediction,
    const int samples, const int t_steps, const int displacement_size, const int threads)
{
    // make a copy of prediction at first
    T* pred_copy;
    cudaSafeCall(cudaMalloc(&pred_copy, samples*t_steps*displacement_size*sizeof(T)));
    cudaSafeCall(cudaMemcpy(pred_copy, prediction, samples*t_steps*displacement_size*sizeof(T), cudaMemcpyDeviceToDevice));

    dim3 blocks{1, samples, t_steps};
    dim3 threads_per_block{threads, 1, 1};

    // call the transpose kernel
    tensor3d_transpose_102_kernel<T><<<blocks, threads_per_block>>> (prediction, pred_copy, t_steps, samples, displacement_size);
    cudaCheckError("tensor3d_transpose_102_kernel error");

    // free the temporary data
    cudaSafeCall(cudaFree(pred_copy));

    // all done
}

template <typename T>
// WARNING still contains wrong case that the Green's functions is multiplying the velocities
// (rather than the cumulative displacement)
void compute_displacement_impl2(cublasHandle_t handle,
    T* predictions, // [samples, t_steps, displacement_size],
    const T* sim_state, //[samples, t_steps, 4*patches]
    const T* gf, // [t_steps, slip_size, displacement_size], input Green's function
    const int samples, // samples or systems
    const int t_steps, // time steps
    const int patches, //  patches * slip_components
    const int displacement_size, // stations*displacement_components
    const T v_0, // v_0
    const T gemm_alpha, // should be 1
    const T gemm_beta  // 0 or -1 (if d_obs is copied to predictions to compute residue)
    )
{
    // allocate slip rate [t_steps, samples, 2*patches]
    auto system_size = patches * 4; // sim_state size per system per t_step
    auto slip_size = patches * 2; // slip rate size per system per t_step
    T* slip_rate;
    cudaSafeCall(cudaMallocManaged(&slip_rate, samples*t_steps*slip_size*sizeof(T)));

    // extract slip rate from sim_state
    extract_slip_rate_transpose<T>(slip_rate, sim_state, samples, t_steps, patches, v_0);


    // Batched Gemm for d_pred = G * M
    // gf [t_steps, slip_size, displacement_size], input Green's function
    // M : slip_rate, been transposed to [t_steps, samples, slip_size]
    // d_pred will be [t_steps, samples, displacement_size], to be tranposed after
    // batches = t_steps, for each batch
    // slip_rate [samples, slip_size] * G [slip_size, displacement_size] = displacement_rate [samples, displacement_size]
    // cublas use Column-major, which is equivalent to
    //  G [displacement_size, slip_size] * slip[slip_size, samples] = displacement_rate [displacement_size, samples]
    cublasSafeCall(cuBLASGemmStridedBatched<T>(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        displacement_size, samples, slip_size, // m, n, k
        &gemm_alpha, // 1
        gf, displacement_size, slip_size*displacement_size, // A, lda, strideA
        slip_rate, slip_size, slip_size*samples, // B, ldc, strideB
        &gemm_beta, // 0 or -1
        predictions, displacement_size, displacement_size*samples,
        t_steps
    ));

    // transpose d_pred from t_steps, samples, displacement_size]
    //   to d_pred [samples, t_steps, displacement_size] for likelihood computation
    transpose_displacement<T>(predictions, samples, t_steps, displacement_size);

    // free slip_rate
    cudaSafeCall(cudaFree(slip_rate));

    // all done
}

// Implementation 3 - assume Cd matrix is different for different t_step: Cd[t_steps, displacement_size, displacement_size]
// but a full Cd with t dependnce will be too big to fit in the GPU memory
// need to make iterations on host/cpu
// TO BE IMPLEMENTED


// cuda kernel to add farfield effects
template <typename T>
__global__ void add_farfield_effects_kernel(
    T* obs_disp,
    const T* obs_farfield,
    const int num_t_obs,
    const int observations) // = 3 * num_stations
{
    // get system index
    int system = blockIdx.x;
    // iterate over all times
    for (auto i_t = threadIdx.x; i_t < num_t_obs; i_t += blockDim.x)
    {
        // iterate over components and stations
        for (auto i_obs = 0; i_obs < observations; i_obs++)
        {
            obs_disp[system * num_t_obs * observations
                     + i_t * observations
                     + i_obs] += obs_farfield[i_t * observations + i_obs];
        }
    }
    // all done
}

// cuda funtion to add farfield effects
template<typename T>
void add_farfield_effects(
    T* obs_disp,
    const T* obs_farfield,
    const int num_systems,
    const int num_t_obs,
    const int num_stations,
    const int threads)
{
    add_farfield_effects_kernel<<<num_systems, threads>>>(
        obs_disp, obs_farfield, num_t_obs, 3 * num_stations);
    cudaCheckError("add_farfield_effects_kernel");
}


// cuda kernel to subtract displacements from the values at t_eq
//
template <typename T>
__global__ void subtract_displacement_from_teq_kernel (T*  obs_disp,
     const int num_systems, const int num_t_obs, const int observations, // observations = 3*num_stations
     const int * i_slips_obs, // indices of t_eq inside t_obs
     const int n_slips_obs)
{
    // get sample/system index
    int system = blockIdx.x;
    // iterate over observations
    for (auto i_obs = threadIdx.x; i_obs < observations; i_obs += blockDim.x)
    {
        // iterate over t_eq
        for(auto i_t_eq = 0; i_t_eq < n_slips_obs; i_t_eq++)
        {
            // get the start index
            auto it_start = i_slips_obs[i_t_eq];
            // get the end index (+1)
            auto it_end = (i_t_eq == n_slips_obs-1) ? num_t_obs : i_slips_obs[i_t_eq+1];
            // get obs_disp value at i_t_eq
            auto offset = obs_disp[(system*num_t_obs+it_start)*observations+i_obs];
            // iterate over all time points
            for(auto it = it_start; it < it_end; it++)
                obs_disp[(system*num_t_obs+it)*observations+i_obs] -= offset;
        }
    }
    // all done
}

// subtract surface displacements from t=t_eq
//
template<typename T>
void subtract_displacement_from_teq(
     T*  obs_disp, // (num_systems, num_t_obs, 3*num_stations)
     const int num_systems, const int num_t_obs, const int observations, // observations = 3*num_stations
     const int * i_slips_obs, // indices of t_eq inside t_obs
     const int n_slips_obs, // total number of equations in t_obs = n_slips_obs
     const int threads
)
{

    subtract_displacement_from_teq_kernel<<<num_systems, threads>>>(
        obs_disp, num_systems, num_t_obs, observations,
        i_slips_obs, n_slips_obs);
    cudaCheckError("subtract_displacement_from_teq_kernel");
}


// cuda kernel to calculate reference displacement timeseries
template <typename T>
__global__ void calculate_ref_timeseries_kernel (
    T* obs_disp,
    const int num_t_obs,
    const int num_stations,
    const bool* obs_mask,
    const int* i_stat_ref,
    const int n_stat_ref,
    T* ref_obs) // (num_systems * num_t_obs * 3)
{
    // get sample/system index
    int system = blockIdx.x;

    // iterate over timestamps and components
    for (auto i_t_comp = threadIdx.x; i_t_comp < num_t_obs * 3; i_t_comp += blockDim.x)
    {
        // accumulate reference station values
        auto i_ref_obs = system * num_t_obs * 3 + i_t_comp;
        ref_obs[i_ref_obs] = 0;
        for (auto ii_stat = 0; ii_stat < n_stat_ref; ii_stat++)
        {
            auto i_stat = i_stat_ref[ii_stat];
            assert(obs_mask[i_t_comp * num_stations + i_stat]);
            ref_obs[i_ref_obs] += obs_disp[system * num_t_obs * 3 * num_stations
                                           + i_t_comp * num_stations + i_stat];
        }
        // divide to get mean
        ref_obs[i_ref_obs] /= n_stat_ref;
    }
}


// cuda kernel to remove reference displacement timeseries
template <typename T>
__global__ void remove_ref_timeseries_kernel (
    T* obs_disp,
    const int num_t_obs,
    const int num_stations,
    T* ref_obs) // (num_systems * num_t_obs * 3)
{
    // get sample/system index
    int system = blockIdx.x;

    // iterate over timestamps and components
    for (auto i_t_comp = threadIdx.x; i_t_comp < num_t_obs * 3; i_t_comp += blockDim.x)
    {
        // iterate over all stations
        for (auto i_stat = 0; i_stat < num_stations; i_stat++)
        {
            // remove reference value
            obs_disp[system * num_t_obs * 3 * num_stations
                     + i_t_comp * num_stations + i_stat] -= ref_obs[system * num_t_obs * 3 + i_t_comp];
        }
    }
}


// cuda kernel to reset to zero after every event, considering data masks
template <typename T>
__global__ void subtract_displacement_from_teq_masked_kernel(
    T* obs_disp,
    const int num_systems,
    const int num_t_obs,
    const int observations, // observations = 3*num_stations
    const int * i_slips_obs, // indices of t_eq inside t_obs
    const int n_slips_obs,
    const bool* obs_mask)
{
    // get sample/system index
    int system = blockIdx.x;
    // iterate over observations
    for (auto i_obs = threadIdx.x; i_obs < observations; i_obs += blockDim.x)
    {
        // iterate over t_eq
        for(auto i_t_eq = 0; i_t_eq < n_slips_obs; i_t_eq++)
        {
            // get the start index
            auto it_start = i_slips_obs[i_t_eq];
            // get the end index (+1)
            auto it_end = (i_t_eq == n_slips_obs-1) ? num_t_obs : i_slips_obs[i_t_eq+1];
            // get obs_disp value at i_t_eq
            // declare offset but keep it undefined until we find the first valid observation
            T offset;
            bool offset_found = false;
            // iterate over all time points
            for(auto it = it_start; it < it_end; it++)
            {
                // skip if the observation isn't valid
                if (obs_mask[it * observations + i_obs])
                {
                    // if this is the first observation after an event, get current offset
                    if (!offset_found)
                    {
                        offset = obs_disp[(system * num_t_obs + it) * observations + i_obs];
                        offset_found = true;
                    }

                    // remove offset
                    obs_disp[(system * num_t_obs + it) * observations + i_obs] -= offset;
                }
            }
        }
    }
    // all done
}


// reference all observations to a set of reference stations,
// remove that displacement from everyone,
// then reset to zero at the first valid observation past an event
template<typename T>
void reference_subtract_reset_displacements(
    T* obs_disp, // (num_systems, num_t_obs, 3*num_stations)
    const int num_systems,
    const int num_t_obs,
    const int num_stations,
    const int* i_slips_obs,
    const int n_slips_obs,
    const bool* obs_mask,  // (num_t_obs, 3*num_stations)
    const int* i_stat_ref,
    const int n_stat_ref,
    const int threads
)
{

    // calculate and remove reference timeseries only if they are provided
    if (n_stat_ref > 0) {
        // allocate reference timeseries
        T* ref_obs;
        cudaSafeCall(cudaMallocManaged(&ref_obs, num_systems * num_t_obs * 3 * sizeof(T)));

        // calculate reference timeseries
        calculate_ref_timeseries_kernel<<<num_systems, threads>>>(
            obs_disp, num_t_obs, num_stations, obs_mask, i_stat_ref, n_stat_ref, ref_obs);
        cudaCheckError("calculate_ref_timeseries_kernel");
        cudaDeviceSynchronize();

        // remove reference timeseries from all stations
        remove_ref_timeseries_kernel<<<num_systems, threads>>>(
            obs_disp, num_t_obs, num_stations, ref_obs);
        cudaCheckError("remove_ref_timeseries_kernel");
        cudaDeviceSynchronize();
    }

    // reset the observations to zero at the first valid observation after an event
    subtract_displacement_from_teq_masked_kernel<<<num_systems, threads>>>(
        obs_disp, num_systems, num_t_obs, 3 * num_stations,
        i_slips_obs, n_slips_obs, obs_mask);
    cudaCheckError("subtract_displacement_from_teq_masked_kernel");
    cudaDeviceSynchronize();
}


} // end of namespace

#endif
// end of file
