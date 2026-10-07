// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//

// declarations
#include "cudaMogi.h"
// the point source, shared with the cpu build
#include <altar/models/mogi/mogi.h>

namespace cudaMogi_kernels {

    using namespace altar::models::mogi;

    // one thread per (sample, observation)
    template <typename real_type>
    __global__ void
    _displacements(matrix_view_t<real_type> theta, matrix_view_t<real_type> stations,
                   std::size_t xIdx, std::size_t yIdx, std::size_t dIdx, std::size_t sIdx,
                   bool log10dV, real_type nu, int batch,
                   matrix_view_t<real_type, false> predicted)
    {
        int observations = stations.packing().shape()[0];
        int idx = blockIdx.x*blockDim.x + threadIdx.x;
        if (idx >= batch * observations) return;
        int sample = idx / observations;
        int obs = idx % observations;

        auto s = theta[{ sample, static_cast<int>(sIdx) }];
        auto dV = log10dV ? pow(real_type(10), s) : s;
        auto u = los<real_type>(
            theta[{ sample, static_cast<int>(xIdx) }], theta[{ sample, static_cast<int>(yIdx) }],
            theta[{ sample, static_cast<int>(dIdx) }], dV, nu,
            stations[{ obs, X }], stations[{ obs, Y }],
            stations[{ obs, LOS_E }], stations[{ obs, LOS_N }], stations[{ obs, LOS_U }]);
        // shift by the offset of the observation's dataset, if any
        auto offset = stations[{ obs, OFFSET }];
        if (offset >= 0) {
            u -= theta[{ sample, static_cast<int>(offset) }];
        }
        predicted[{ sample, obs }] = u;
    }

} // of namespace cudaMogi_kernels


template <typename real_type>
void altar::models::mogi::cuda::
displacements(matrix_view_t<real_type> theta, matrix_view_t<real_type> stations,
              std::size_t xIdx, std::size_t yIdx, std::size_t dIdx, std::size_t sIdx,
              bool log10dV, real_type nu, std::size_t batch,
              matrix_view_t<real_type, false> predicted, cudaStream_t stream)
{
    int observations = stations.packing().shape()[0];
    int threads = static_cast<int>(batch) * observations;
    if (threads == 0) return;
    cudaMogi_kernels::_displacements<real_type><<<IDIVUP(threads, NTHREADS), NTHREADS, 0, stream>>>(
        theta, stations, xIdx, yIdx, dIdx, sIdx, log10dV, nu, static_cast<int>(batch), predicted);
    cudaCheckError("cudaMogi::displacements error");
}

template void altar::models::mogi::cuda::displacements<float>(
    matrix_view_t<float>, matrix_view_t<float>, std::size_t, std::size_t, std::size_t, std::size_t,
    bool, float, std::size_t, matrix_view_t<float, false>, cudaStream_t);
template void altar::models::mogi::cuda::displacements<double>(
    matrix_view_t<double>, matrix_view_t<double>, std::size_t, std::size_t, std::size_t, std::size_t,
    bool, double, std::size_t, matrix_view_t<double, false>, cudaStream_t);

// end of file
