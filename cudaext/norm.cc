// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-2019 parasim inc
// (c) 2010-2019 california institute of technology
// all rights reserved
//
// Author(s): Lijun Zhu

#include <portinfo>
#include <Python.h>
#include <cmath>
#include <iostream>
#include <iomanip>

// c++ class includes
#include <altar/cuda/norm/cudaL2.h>

// cuda utilities
#include <pyre/cuda.h>
#include <pyre/cuda/cudaext.h>

// my declaration
#include "norm.h"
#include "capsules.h"


// cudaL2_norm ||x||
const char * const altar::cuda::extensions::cudaL2::norm__name__ = "cudaL2_norm";
const char * const altar::cuda::extensions::cudaL2::norm__doc__ =
    "cudaL2 norm of a batch of data";
    
PyObject *
altar::cuda::extensions::cudaL2::norm(PyObject *, PyObject * args) {
    // the args 
    // norm(data, , batch) batch <= samples
    // data (samples, parameters) proba
    PyObject * dataCapsule, *resultCapsule;  
    size_t batch;
    
    int status = PyArg_ParseTuple(args, "O!O!k:cudaL2_norm", 
                                    &PyCapsule_Type, &dataCapsule, &PyCapsule_Type, &resultCapsule,
                                    &batch);
    if(!status) return 0;
    if (!PyCapsule_IsValid(dataCapsule, altar::cuda::extensions::matrix::capsule_t) ||
        !PyCapsule_IsValid(resultCapsule, altar::cuda::extensions::vector::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid matrix/vector capsule");
        return 0;
    }

    cuda_matrix * data = static_cast<cuda_matrix *>
        (PyCapsule_GetPointer(dataCapsule, altar::cuda::extensions::matrix::capsule_t));
    cuda_vector * result = static_cast<cuda_vector *>
        (PyCapsule_GetPointer(resultCapsule, altar::cuda::extensions::vector::capsule_t));

    //       void norm(const real_type * const data, real_type * const probability, 
    //                const size_t batch, const size_t parameters, cudaStream_t stream=0);
    size_t parameters = data->size2;
    switch(data->dtype) {
        case PYCUDA_DOUBLE:
            altar::cuda::norms::cudaL2::norm<double>((double *)data->data, (double *)result->data, batch, parameters);
            break;
        case PYCUDA_FLOAT:
            altar::cuda::norms::cudaL2::norm<float>((float *)data->data, (float *)result->data, batch, parameters);
            break;
        default:
            PyErr_SetString(PyExc_TypeError, "not a real type");
    }
    
    // all done
    // return None                                                                                                                             
    Py_INCREF(Py_None);
    return Py_None;
}


// cudaL2_normllk l2const - ||x||^2/2
const char * const altar::cuda::extensions::cudaL2::normllk__name__ = "cudaL2_normllk";
const char * const altar::cuda::extensions::cudaL2::normllk__doc__ =
    "cudaL2 normllk of a batch of data";
    
PyObject *
altar::cuda::extensions::cudaL2::normllk(PyObject *, PyObject * args) {
    // the args 
    // normllk(data, , batch) batch <= samples
    // data (samples, parameters) proba
    PyObject * dataCapsule, *resultCapsule;  
    size_t batch;
    double l2constant;
    
    int status = PyArg_ParseTuple(args, "O!O!kd:cudaL2_normllk", 
                                    &PyCapsule_Type, &dataCapsule, &PyCapsule_Type, &resultCapsule,
                                    &batch, &l2constant);
    if(!status) return 0;
    if (!PyCapsule_IsValid(dataCapsule, altar::cuda::extensions::matrix::capsule_t) ||
        !PyCapsule_IsValid(resultCapsule, altar::cuda::extensions::vector::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid matrix/vector capsule");
        return 0;
    }

    cuda_matrix * data = static_cast<cuda_matrix *>
        (PyCapsule_GetPointer(dataCapsule, altar::cuda::extensions::matrix::capsule_t));
    cuda_vector * result = static_cast<cuda_vector *>
        (PyCapsule_GetPointer(resultCapsule, altar::cuda::extensions::vector::capsule_t));
/*
template <typename real_type>
void altar::cuda::norms::cudaL2::
normllk(const real_type* const data, // input data , matrix(samples, parameters)
    real_type* const probability, // output norm, vector(samples)
    const size_t batch, // first batch of samples to be computed batch<=samples
    const size_t parameters, // number of parameters
    const real_type l2constant, // constant to be added to probability
    cudaStream_t stream)
*/
    size_t parameters = data->size2;
    switch(data->dtype) {
        case PYCUDA_DOUBLE:
            altar::cuda::norms::cudaL2::normllk<double>((double *)data->data, (double *)result->data, batch, parameters, l2constant);
            break;
        case PYCUDA_FLOAT:
            altar::cuda::norms::cudaL2::normllk<float>((float *)data->data, (float *)result->data, batch, parameters, (float)l2constant);
            break;
        default:
            PyErr_SetString(PyExc_TypeError, "not a real type");
    }
    
    // all done
    // return None                                                                                                                             
    Py_INCREF(Py_None);
    return Py_None;
}
// end of file
