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
#include "moment.h"

// c++ class includes
#include <altar/models/seismic/cuda/cudaMoment.h>

// local includes
#include "capsules.h"

// cuda utilities
#include <pyre/cuda.h>
#include <pyre/cuda/capsules.h>



const char * const altar::extensions::models::cudaseismic::moment_logpdf__name__ = "cudaMoment_logpdf";
const char * const altar::extensions::models::cudaseismic::moment_logpdf__doc__ =
    "cudaMoment compute log pdf of the total moment";

PyObject *
altar::extensions::models::cudaseismic::moment_logpdf(PyObject *, PyObject * args) {
    // the arguments
    // sample(theta, probability, samples, (idx_begin, idx_end), (mean, sigma), mu_area)

    PyObject * thetaCapsule, * probabilityCapsule;
    size_t idx_begin, idx_end; // parameter index
    double mean, sigma; // support or range
    PyObject * mu_areaCapsule;
    double moment_constraint_factor; // a factor to tune the strength of moment constraint in the logpdf calculation
    size_t samples;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
        args, "O!O!k(kk)(dd)O!d:cudaMoment_logpdf",
        &PyCapsule_Type, &thetaCapsule,
        &PyCapsule_Type, &probabilityCapsule,
        &samples, &idx_begin, &idx_end,
        &mean, &sigma,
        &PyCapsule_Type, &mu_areaCapsule,
        &moment_constraint_factor 
        );
    // if something went wrong
    if (!status) return 0;
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(thetaCapsule, matrix::capsule_t)
            || !PyCapsule_IsValid(probabilityCapsule, vector::capsule_t) 
            || !PyCapsule_IsValid(mu_areaCapsule, vector::capsule_t))
    {
        PyErr_SetString(PyExc_TypeError, "invalid capsule for cudaMoment_logpdf");
        return 0;
    }

    // convert PyObjects to C Objects
    cuda_matrix * theta = static_cast<cuda_matrix *>
        (PyCapsule_GetPointer(thetaCapsule, matrix::capsule_t));
    cuda_vector * prob = static_cast<cuda_vector *>
        (PyCapsule_GetPointer(probabilityCapsule, vector::capsule_t));
    cuda_vector * mu_area = static_cast<cuda_vector *> 
        (PyCapsule_GetPointer(mu_areaCapsule, vector::capsule_t));    
    size_t parameters = theta->size2;

    // call c method
    /* template <typename real_type>
        void altar::models::seismic::moment_
        logpdf(const real_type * const theta, real_type * const probability,
                    const size_t samples, const size_t parameters,
                    const size_t idx_begin, const size_t idx_end,
                    const real_type mean, const real_type sigma,
                    const real_type * const mu_area,
                    const real_type moment_constraint_factor,
                    cudaStream_t stream)
    */
    if(theta->dtype == PYCUDA_FLOAT) //single precision
    {
        altar::models::seismic::cudaMoment::logpdf<float>
            ((const float *)theta->data, (float *)prob->data,
            samples, parameters, idx_begin, idx_end, (float)mean, (float)sigma, (const float *)mu_area->data, (float)moment_constraint_factor);
    }
    else //double precision
    {
        altar::models::seismic::cudaMoment::logpdf<double>
            ((const double *)theta->data, (double *)prob->data,
            samples, parameters, idx_begin, idx_end, mean, sigma, (const double *)mu_area->data, moment_constraint_factor);
    }
    // all done
    // return None
    Py_INCREF(Py_None);
    return Py_None;
}

// end of file
