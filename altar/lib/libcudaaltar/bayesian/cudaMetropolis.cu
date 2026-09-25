// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//
// hailiang zhang
// declarations
#include "cudaMetropolis.h"
// cuda utitlities
// shared NTHREADS/IDIVUP/cudaCheckError/matrix_view_t/vector_view_t
#include "../support.h"

#include <pyre/cuda.h>
#include <iostream>

// cuda kernels; defined here, ahead of the launcher functions below that instantiate and
// launch them (see cudaL2.cu/cudaGaussian.cu for why the ordering matters)
namespace cudaMetropolis_kernels {

    // one thread per sample: if {invalid[id]} is false, atomically claim the next slot in
    // {valid_sample_indices} and record {id} there -- {valid_samples[0]} (zeroed by the
    // caller first) ends up holding the total count
    __global__ void
    _setValidSampleIndices(vector_view_t<int> valid_sample_indices, vector_view_t<int, true> invalid,
        vector_view_t<int> valid_samples)
    {
        int id = blockIdx.x*blockDim.x + threadIdx.x;
        auto samples = valid_sample_indices.packing().shape()[0];
        if (id >= samples) return;

        if (!invalid[{ id }]) {
            auto index_to_fill = atomicAdd(&valid_samples[{ 0 }], 1);
            valid_sample_indices[{ index_to_fill }] = id;
        }
    }

    // gather: theta_candidate[sample, :] = theta_proposal[valid_sample_indices[sample], :],
    // one thread per (sample, parameter) pair
    template <typename realtype_t>
    __global__ void
    _queueValidSamples(matrix_view_t<realtype_t, false> theta_candidate,
        matrix_view_t<realtype_t, true> theta_proposal,
        vector_view_t<int, true> valid_sample_indices,
        const size_t samples)
    {
        int sample = blockIdx.x*blockDim.x + threadIdx.x;
        int parameter = blockIdx.y*blockDim.y + threadIdx.y;
        auto parameters = theta_candidate.packing().shape()[1];
        if (sample >= samples || parameter >= parameters) return;

        theta_candidate[{ sample, parameter }] =
            theta_proposal[{ valid_sample_indices[{ sample }], parameter }];
    }

    // one Metropolis-Hastings accept/reject test per valid sample
    template <typename realtype_t>
    __global__ void
    _metropolisUpdate(matrix_view_t<realtype_t, false> theta,
        vector_view_t<realtype_t, false> prior,
        vector_view_t<realtype_t, false> data,
        vector_view_t<realtype_t, false> posterior,
        matrix_view_t<realtype_t, true> theta_candidate,
        vector_view_t<realtype_t, true> prior_candidate,
        vector_view_t<realtype_t, true> data_candidate,
        vector_view_t<realtype_t, true> posterior_candidate,
        vector_view_t<realtype_t, true> dices,
        vector_view_t<int> acceptance_flag,
        vector_view_t<int, true> valid_sample_indices,
        const size_t batch)
    {
        int sample = blockIdx.x*blockDim.x + threadIdx.x;
        if (sample >= batch) return;

        auto parameters = theta.packing().shape()[1];
        int sample_index = valid_sample_indices[{ sample }];

        if (log(dices[{ sample }]) <= posterior_candidate[{ sample }] - posterior[{ sample_index }]) {
            // acceptance: copy theta
            for (int parameter = 0; parameter < parameters; ++parameter) {
                theta[{ sample_index, parameter }] = theta_candidate[{ sample, parameter }];
            }
            // copy densities
            prior[{ sample_index }] = prior_candidate[{ sample }];
            data[{ sample_index }] = data_candidate[{ sample }];
            posterior[{ sample_index }] = posterior_candidate[{ sample }];
            // set the flag
            acceptance_flag[{ sample }] = 1;
        }
    }

} // of namespace cudaMetropolis_kernels

// launch {cudaMetropolis_kernels::_setValidSampleIndices}
void altar::cuda::bayesian::cudaMetropolis::
setValidSampleIndices(vector_view_t<int> valid_sample_indices, vector_view_t<int, true> invalid,
    vector_view_t<int> valid_samples,
    cudaStream_t stream)
{
    auto samples = valid_sample_indices.packing().shape()[0];
    auto blockSize = NTHREADS;
    auto gridSize = IDIVUP(samples, blockSize);

    cudaSafeCall(cudaMemsetAsync(valid_samples.data(), 0, sizeof(int), stream));
    cudaSafeCall(cudaMemsetAsync(valid_sample_indices.data(), 0, samples*sizeof(int), stream));
    cudaMetropolis_kernels::_setValidSampleIndices<<<gridSize, blockSize, 0, stream>>>(
        valid_sample_indices, invalid, valid_samples);
    cudaCheckError("cudaMetropolis::setValidSampleIndices error");
}


// launch {cudaMetropolis_kernels::_queueValidSamples}
template <typename realtype_t>
void altar::cuda::bayesian::cudaMetropolis::
queueValidSamples(matrix_view_t<realtype_t, false> theta_candidate,
    matrix_view_t<realtype_t, true> theta_proposal,
    vector_view_t<int, true> valid_sample_indices,
    const size_t samples,
    cudaStream_t stream)
{
    auto parameters = theta_candidate.packing().shape()[1];
    // one thread per (sample, parameter) pair
    dim3 blockSize(BDIMX, BDIMY, 1);
    dim3 gridSize(IDIVUP(samples, blockSize.x), IDIVUP(parameters, blockSize.y), 1);

    cudaMetropolis_kernels::_queueValidSamples<realtype_t><<<gridSize, blockSize, 0, stream>>>(
        theta_candidate, theta_proposal, valid_sample_indices, samples);
    cudaCheckError("cudaMetropolis::queueValidSamples error");
}

template void altar::cuda::bayesian::cudaMetropolis::queueValidSamples<float>(
    matrix_view_t<float, false>, matrix_view_t<float, true>, vector_view_t<int, true>, const size_t, cudaStream_t);
template void altar::cuda::bayesian::cudaMetropolis::queueValidSamples<double>(
    matrix_view_t<double, false>, matrix_view_t<double, true>, vector_view_t<int, true>, const size_t, cudaStream_t);


// launch {cudaMetropolis_kernels::_metropolisUpdate}
template <typename realtype_t>
void altar::cuda::bayesian::cudaMetropolis::
metropolisUpdate(matrix_view_t<realtype_t, false> theta,
    vector_view_t<realtype_t, false> prior,
    vector_view_t<realtype_t, false> data,
    vector_view_t<realtype_t, false> posterior,
    matrix_view_t<realtype_t, true> theta_candidate,
    vector_view_t<realtype_t, true> prior_candidate,
    vector_view_t<realtype_t, true> data_candidate,
    vector_view_t<realtype_t, true> posterior_candidate,
    vector_view_t<realtype_t, true> dices,
    vector_view_t<int> acceptance_flag,
    vector_view_t<int, true> valid_sample_indices,
    const size_t batch,
    cudaStream_t stream)
{
    auto blockSize = NTHREADS;
    auto gridSize = IDIVUP(batch, blockSize);

    cudaMetropolis_kernels::_metropolisUpdate<realtype_t><<<gridSize, blockSize, 0, stream>>>(
        theta, prior, data, posterior,
        theta_candidate, prior_candidate, data_candidate, posterior_candidate,
        dices, acceptance_flag, valid_sample_indices,
        batch);
    cudaCheckError("cudaMetropolis::metropolisUpdate error");
}

template void altar::cuda::bayesian::cudaMetropolis::metropolisUpdate<float>(
    matrix_view_t<float, false>, vector_view_t<float, false>, vector_view_t<float, false>, vector_view_t<float, false>,
    matrix_view_t<float, true>, vector_view_t<float, true>, vector_view_t<float, true>, vector_view_t<float, true>,
    vector_view_t<float, true>, vector_view_t<int>, vector_view_t<int, true>, const size_t, cudaStream_t);
template void altar::cuda::bayesian::cudaMetropolis::metropolisUpdate<double>(
    matrix_view_t<double, false>, vector_view_t<double, false>, vector_view_t<double, false>, vector_view_t<double, false>,
    matrix_view_t<double, true>, vector_view_t<double, true>, vector_view_t<double, true>, vector_view_t<double, true>,
    vector_view_t<double, true>, vector_view_t<int>, vector_view_t<int, true>, const size_t, cudaStream_t);

// end of file
