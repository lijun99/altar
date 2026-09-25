// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//
// hailiang zhang
// declarations
#include "cudaL2.h"

// cuda utilities
// shared NTHREADS/IDIVUP/cudaCheckError
#include "../support.h"

#include <pyre/cuda.h>

// cuda kernels; defined here, ahead of the launcher functions below that instantiate and
// launch them -- nvcc's handling of a __global__ function template instantiated from a
// launch that only sees a forward declaration (body defined later in the same file) is not
// reliable the way a plain host template's deferred instantiation is, so the definition has
// to come first
namespace cudaL2_kernels {
    // one thread per sample: probability[sample] = ||data[sample, :]||, the l2 norm of that
    // sample's row of {data}
    template <typename real_type>
    __global__ void
    _norm(altar::cuda::norms::cudaL2::data_view_t<real_type> data,
        altar::cuda::norms::cudaL2::result_view_t<real_type> probability,
        size_t batch)
    {
        // {blockIdx.x*blockDim.x + threadIdx.x} is {unsigned int}, not deducible with {auto}
        // here: it must stay a *signed* type, since indexing a grid ({data[{sample, i}]}
        // below) builds a {pyre::grid::Index} from a braced-init-list, and list-init narrows
        // an unsigned source to the (signed) index type, which the compiler rejects outright
        int sample = blockIdx.x*blockDim.x + threadIdx.x;
        if (sample >= batch) return;

        // the extent travels with the grid, no separate {parameters} argument needed
        auto parameters = data.packing().shape()[1];

        auto prob = real_type{ 0 };
        for (auto i = 0; i < parameters; ++i)
        {
            auto value = data[{ sample, i }];
            prob += value*value;
        }

        probability[{ sample }] = sqrt(prob);
    }

    // one thread per sample: probability[sample] = l2constant - 0.5 * ||data[sample, :]||^2,
    // the l2 log likelihood of that sample's row of {data}
    template <typename real_type>
    __global__ void
    _normllk(altar::cuda::norms::cudaL2::data_view_t<real_type> data,
        altar::cuda::norms::cudaL2::result_view_t<real_type> probability,
        const size_t batch,
        const real_type l2constant)
    {
        // see {_norm}: {sample} must stay signed, so not {auto}
        int sample = blockIdx.x*blockDim.x + threadIdx.x;
        if (sample >= batch) return;

        auto parameters = data.packing().shape()[1];

        auto prob = real_type{ 0 };
        for (auto i = 0; i < parameters; ++i)
        {
            auto value = data[{ sample, i }];
            prob += value*value;
        }

        probability[{ sample }] = l2constant - 0.5*prob;
    }
} // of namespace cudaL2_kernels

// launch {cudaL2_kernels::_norm}: fill {probability} with the l2 norm of the first {batch}
// rows of {data}, one thread per row
template <typename real_type>
void altar::cuda::norms::cudaL2::
norm(data_view_t<real_type> data, // input data, a (samples x parameters) view
    result_view_t<real_type> probability, // output norm, a (samples,) view
    const size_t batch, // first batch of samples to be computed batch<=samples
    cudaStream_t stream)
{
    // one thread per sample
    auto blockSize = NTHREADS;
    auto gridSize = IDIVUP(batch, blockSize);

    // call cuda kernels
    cudaL2_kernels::_norm<real_type><<<gridSize, blockSize, 0, stream>>>(
        data, probability, batch);
    cudaCheckError("cudaL2::L2norm error");
}

// explicit instantiation
template void altar::cuda::norms::cudaL2::norm<float>(
    data_view_t<float>, result_view_t<float>, const size_t, cudaStream_t);
template void altar::cuda::norms::cudaL2::norm<double>(
    data_view_t<double>, result_view_t<double>, const size_t, cudaStream_t);


// launch {cudaL2_kernels::_normllk}: fill {probability} with {l2constant} - 0.5 * ||data||^2
// for the first {batch} rows of {data}, one thread per row
template <typename real_type>
void altar::cuda::norms::cudaL2::
normllk(data_view_t<real_type> data, // input data, a (samples x parameters) view
    result_view_t<real_type> probability, // output norm, a (samples,) view
    const size_t batch, // first batch of samples to be computed batch<=samples
    const real_type l2constant, // constant to be added to probability
    cudaStream_t stream)
{
    // one thread per sample
    auto blockSize = NTHREADS;
    auto gridSize = IDIVUP(batch, blockSize);

    // call cuda kernels
    cudaL2_kernels::_normllk<real_type><<<gridSize, blockSize, 0, stream>>>(
        data, probability, batch, l2constant);
    cudaCheckError("cudaL2::L2normLLK error");
}

// explicit instantiation
template void altar::cuda::norms::cudaL2::normllk<float>(
    data_view_t<float>, result_view_t<float>, const size_t, const float, cudaStream_t);
template void altar::cuda::norms::cudaL2::normllk<double>(
    data_view_t<double>, result_view_t<double>, const size_t, const double, cudaStream_t);

// end of file
