// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//

// code guard
#ifndef altar_cuda_grids_cudaGrids_h
#define altar_cuda_grids_cudaGrids_h

#include <cstddef>

// place everything in the local namespace
namespace altar {
    namespace cuda {
        namespace grids {
            // the sum of {size} cells at {cells}, on the device, accumulated in {accumulator_t}
            template <typename cell_t, typename accumulator_t>
            auto sum(const cell_t * cells, std::size_t size) -> accumulator_t;
        } // of namespace grids
    } // of namespace cuda
} // of namespace altar

#endif

// end of file
