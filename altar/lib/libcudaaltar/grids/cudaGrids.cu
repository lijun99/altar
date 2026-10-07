// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//

// externals
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
// my declarations
#include "cudaGrids.h"


// the default stream orders the reduction after the kernels that wrote the cells
template <typename cell_t, typename accumulator_t>
auto
altar::cuda::grids::sum(const cell_t * cells, std::size_t size) -> accumulator_t
{
    return thrust::reduce(thrust::device, cells, cells + size, accumulator_t(0));
}


// explicit instantiations
template auto altar::cuda::grids::sum<int, long long>(const int *, std::size_t) -> long long;
template auto altar::cuda::grids::sum<long long, long long>(const long long *, std::size_t) -> long long;
template auto altar::cuda::grids::sum<float, double>(const float *, std::size_t) -> double;
template auto altar::cuda::grids::sum<double, double>(const double *, std::size_t) -> double;

// end of file
