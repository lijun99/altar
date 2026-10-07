// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2022-2023 california institute of technology
// all rights reserved


// code guard
#ifndef __cuda_ode_external_h__
#define __cuda_ode_external_h__


// cuda
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include <cuda/std/limits>
#include <cuda/std/cmath>


// cooperative groups
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

//
#include "error.h"

#define __ALIGNED__ __align__(16)
#define NTHREADS 256 // default number of threads per block

#endif //__cuda_ode_external_h__
// end of file
