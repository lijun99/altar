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

// c++ class includes
#include <altar/cuda/bayesian/cudaLangevin.h>

// cuda utilities
#include <pyre/cuda.h>
#include <pyre/cuda/capsules.h>

// my declaration
#include "langevin.h"
#include "capsules.h"

// Langevin update theta for a particular parameter
const char * const altar::cuda::extensions::cudaLangevin::updateTheta__name__ = "cudaLangevin_updateTheta";
const char * const altar::cuda::extensions::cudaLangevin::updateTheta__doc__ =
    "cudaLangevin accept/reject procedure and update original state with accepted samples";


PyObject *
altar::cuda::extensions::cudaLangevin::updateTheta(PyObject *, PyObject * args)
{
    PyObject *thetaCapsule, *priorCapsule, *dataCapsule, *etaCapsule;
    double half_epsilon_t;
    size_t index;

    int status = PyArg_ParseTuple(args, "O!O!O!dO!k:cudaLangevin_updateTheta",
                                    &PyCapsule_Type, &thetaCapsule,
                                    &PyCapsule_Type, &priorCapsule,
                                    &PyCapsule_Type, &dataCapsule,
                                    &half_epsilon_t,
                                    &PyCapsule_Type, &etaCapsule,
                                    &index);
    if(!status) return 0;
    // check the capsule types of input
    if (!PyCapsule_IsValid(thetaCapsule, altar::cuda::extensions::matrix::capsule_t) ||
        !PyCapsule_IsValid(priorCapsule, altar::cuda::extensions::vector::capsule_t) ||
        !PyCapsule_IsValid(dataCapsule, altar::cuda::extensions::vector::capsule_t) ||
        !PyCapsule_IsValid(etaCapsule, altar::cuda::extensions::vector::capsule_t))
    {
        PyErr_SetString(PyExc_TypeError, "Langevin::updateTheta invalid matrix/vector capsule");
        return 0;
    }

    // cast capsules to c pointers
    cuda_matrix * theta = static_cast<cuda_matrix *>
        (PyCapsule_GetPointer(thetaCapsule, altar::cuda::extensions::matrix::capsule_t));
    cuda_vector * prior = static_cast<cuda_vector *>
        (PyCapsule_GetPointer(priorCapsule, altar::cuda::extensions::vector::capsule_t));
    cuda_vector * data = static_cast<cuda_vector *>
        (PyCapsule_GetPointer(dataCapsule, altar::cuda::extensions::vector::capsule_t));
    cuda_vector * eta_t = static_cast<cuda_vector *>
        (PyCapsule_GetPointer(etaCapsule, altar::cuda::extensions::vector::capsule_t));

    const size_t samples = theta->size1;
    const size_t parameters = theta->size2;

   /*
    updateTheta(realtype_t * const theta,
                const realtype_t * const prior_gradient,
                const realtype_t * const datalikelihood_gradient,
                const realtype_t half_epsilon_t, const realtype_t * const eta_t,
                const size_t samples, const size_t parameters, const size_t index,
                cudaStream_t stream)
                */
    switch(theta->dtype) {
    case PYCUDA_FLOAT:
        altar::cuda::bayesian::cudaLangevin::updateTheta<float>(
            (float *)theta->data, (float *)prior->data,
            (float *)data->data, (float)half_epsilon_t,
            (const float *)eta_t->data,
            samples, parameters, index);
        break;
    case PYCUDA_DOUBLE:
        altar::cuda::bayesian::cudaLangevin::updateTheta<double>(
            (double *)theta->data, (double *)prior->data,
            (double *)data->data, half_epsilon_t,
            (const double *)eta_t->data,
            samples, parameters, index);
        break;
    default:
        PyErr_SetString(PyExc_TypeError, "invalid datatype: only double/float are supported");
        return 0;
    }
    // all done
    // return none
    Py_RETURN_NONE;
}

// Langevin update theta for all parameters at the same time
const char * const altar::cuda::extensions::cudaLangevin::updateThetaBatched__name__ = "cudaLangevin_updateThetaBatched";
const char * const altar::cuda::extensions::cudaLangevin::updateThetaBatched__doc__ =
    "cudaLangevin accept/reject procedure and update original state with accepted samples";


PyObject *
altar::cuda::extensions::cudaLangevin::updateThetaBatched(PyObject *, PyObject * args)
{
    PyObject *thetaCapsule, *priorCapsule, *dataCapsule, *etaCapsule;
    double half_epsilon_t, alpha1, alpha2;
    size_t batch;

    int status = PyArg_ParseTuple(args, "O!dO!dO!dO!k:cudaLangevin_updateThetaBatched",
                                    &PyCapsule_Type, &thetaCapsule,
                                    &alpha1, &PyCapsule_Type, &priorCapsule,
                                    &alpha2, &PyCapsule_Type, &dataCapsule,
                                    &half_epsilon_t,
                                    &PyCapsule_Type, &etaCapsule,
                                    &batch);
    if(!status) return 0;
    // check the capsule types of input
    if (!PyCapsule_IsValid(thetaCapsule, altar::cuda::extensions::matrix::capsule_t) ||
        !PyCapsule_IsValid(priorCapsule, altar::cuda::extensions::matrix::capsule_t) ||
        !PyCapsule_IsValid(dataCapsule, altar::cuda::extensions::matrix::capsule_t) ||
        !PyCapsule_IsValid(etaCapsule, altar::cuda::extensions::matrix::capsule_t))
    {
        PyErr_SetString(PyExc_TypeError, "Langevin::updateThetaBatched invalid matrix/vector capsule");
        return 0;
    }

    // cast capsules to c pointers
    cuda_matrix * theta = static_cast<cuda_matrix *>
        (PyCapsule_GetPointer(thetaCapsule, altar::cuda::extensions::matrix::capsule_t));
    cuda_matrix * prior = static_cast<cuda_matrix *>
        (PyCapsule_GetPointer(priorCapsule, altar::cuda::extensions::matrix::capsule_t));
    cuda_matrix * data = static_cast<cuda_matrix *>
        (PyCapsule_GetPointer(dataCapsule, altar::cuda::extensions::matrix::capsule_t));
    cuda_matrix * eta_t = static_cast<cuda_matrix *>
        (PyCapsule_GetPointer(etaCapsule, altar::cuda::extensions::matrix::capsule_t));

    const size_t samples = theta->size1;
    const size_t parameters = theta->size2;

    switch(theta->dtype) {
    case PYCUDA_FLOAT:
        altar::cuda::bayesian::cudaLangevin::updateThetaBatched<float>(
            (float *)theta->data,
            (float)alpha1, (float *)prior->data,
            (float)alpha2, (float *)data->data,
            (float)half_epsilon_t,
            (const float *)eta_t->data,
            batch, parameters);
        break;
    case PYCUDA_DOUBLE:
        altar::cuda::bayesian::cudaLangevin::updateThetaBatched<double>(
            (double *)theta->data,
            alpha1, (double *)prior->data,
            alpha2, (double *)data->data,
            half_epsilon_t,
            (const double *)eta_t->data,
            batch, parameters);
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
