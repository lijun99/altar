// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-2021 parasim inc
// (c) 2010-2021 california institute of technology
// all rights reserved
//
// Author(s): Lijun Zhu

#include <portinfo>
#include <Python.h>
#include <cmath>
#include <iostream>
#include <iomanip>


// declarations
#include "static.h"

// c++ class includes
#include <altar/models/seismic/cuda/cudaStatic.h>

// local includes
#include "capsules.h"

// cuda utilities
#include <pyre/cuda.h>
#include <pyre/cuda/capsules.h>


// types
namespace altar {
    namespace extensions {
        namespace models {
            namespace cudaseismic {
                namespace vector = pyre::extensions::cuda::vector;
                namespace matrix = pyre::extensions::cuda::matrix;
                namespace stream = pyre::extensions::cuda::stream;
            }
        }
    }
}



// allocate a cuda/c kinematic model
const char * const altar::extensions::models::cudaseismic::static_gemm_col__name__ = "static_gemm_col";
const char * const altar::extensions::models::cudaseismic::static_gemm_col__doc__ = "gemm_col for gradient calculation";

PyObject *
altar::extensions::models::cudaseismic::
static_gemm_col(PyObject *, PyObject * args)
{
    // parameters
    PyObject *Capsule_A, *Capsule_B, *Capsule_C;
    size_t index;
    double factor;

    int status = PyArg_ParseTuple(args, "O!O!O!kd:static_gemm_col",
                                    &PyCapsule_Type, &Capsule_A,
                                    &PyCapsule_Type, &Capsule_B,
                                    &PyCapsule_Type, &Capsule_C,
                                    &index, &factor);
    if(!status) return 0;
    if (!PyCapsule_IsValid(Capsule_C, vector::capsule_t) ||
        !PyCapsule_IsValid(Capsule_B, matrix::capsule_t) ||
        !PyCapsule_IsValid(Capsule_A, matrix::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid matrix/vector capsule in static_gemm_col");
        return 0;
    }

    cuda_vector * C = static_cast<cuda_vector *>
        (PyCapsule_GetPointer(Capsule_C, vector::capsule_t));
    cuda_matrix * B = static_cast<cuda_matrix *>
        (PyCapsule_GetPointer(Capsule_B, matrix::capsule_t));
    cuda_matrix * A = static_cast<cuda_matrix *>
        (PyCapsule_GetPointer(Capsule_A, matrix::capsule_t));

    const size_t M = A->size1;
    const size_t K = A->size2;
    const size_t N = B->size2;
    switch(A->dtype) {
    case PYCUDA_FLOAT:
        altar::models::seismic::cudaStatic::gemm_col<float>(
            (const float *)A->data, (const float *)B->data,
            (float *) C->data, M, K, N, index, (float)factor);
        break;
    case PYCUDA_DOUBLE:
        altar::models::seismic::cudaStatic::gemm_col<double>(
            (const double *)A->data, (const double *)B->data,
            (double *) C->data, M, K, N, index, factor);
        break;
    default:
        PyErr_SetString(PyExc_TypeError, "invalid datatype: only double/float are supported");
        return 0;
    }
    // all done
    // return none
    Py_RETURN_NONE;
}
// end of file
