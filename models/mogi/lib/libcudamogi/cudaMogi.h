// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//

// code guard
#pragma once

#include <cuda_runtime.h>
// {matrix_view_t}
#include <altar/cuda/support.h>

namespace altar::models::mogi::cuda {

    // fill the first {batch} rows of {predicted} (samples x observations) with the LOS
    // displacements, less the dataset offsets, of the Mogi sources in {theta}; {stations} is laid
    // out as in {altar::models::mogi::station_t}
    template <typename real_type>
    void displacements(matrix_view_t<real_type> theta, matrix_view_t<real_type> stations,
                       std::size_t xIdx, std::size_t yIdx, std::size_t dIdx, std::size_t sIdx,
                       bool log10dV, real_type nu, std::size_t batch,
                       matrix_view_t<real_type, false> predicted, cudaStream_t stream = 0);

} // of namespace altar::models::mogi::cuda

// end of file
