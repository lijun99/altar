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
#include <altar/cuda/bayesian/cudaMetropolis.h>

// cuda utilities
#include <pyre/cuda.h>
#include <pyre/cuda/capsules.h>

// my declaration
#include "metropolis.h"
#include "capsules.h"


// label valid samples
const char * const altar::cuda::extensions::cudaMetropolis::setValidSampleIndices__name__ = "cudaMetropolis_setValidSampleIndices";
const char * const altar::cuda::extensions::cudaMetropolis::setValidSampleIndices__doc__ =
    "cudaMetropolis label valid samples";

PyObject *
altar::cuda::extensions::cudaMetropolis::setValidSampleIndices(PyObject *, PyObject * args)
{
    PyObject * indicesCapsule, *flagsCapsule, *validCountCap;

    int status = PyArg_ParseTuple(args, "O!O!O!:cudaMetropolis_setValidSampleIndices",
                                    &PyCapsule_Type, &indicesCapsule,
                                    &PyCapsule_Type, &flagsCapsule,
                                    &PyCapsule_Type, &validCountCap);
    if(!status) return 0;
    if (!PyCapsule_IsValid(indicesCapsule, altar::cuda::extensions::vector::capsule_t) ||
        !PyCapsule_IsValid(flagsCapsule, altar::cuda::extensions::vector::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid matrix/vector capsule");
        return 0;
    }

    cuda_vector * indices = static_cast<cuda_vector *>
        (PyCapsule_GetPointer(indicesCapsule, altar::cuda::extensions::vector::capsule_t));
    cuda_vector * flags = static_cast<cuda_vector *>
        (PyCapsule_GetPointer(flagsCapsule, altar::cuda::extensions::vector::capsule_t));

    cuda_vector * validCount = static_cast<cuda_vector *>
        (PyCapsule_GetPointer(validCountCap, altar::cuda::extensions::vector::capsule_t));

    /* void altar::cuda::bayesian::cudaMetropolis::
    setValidSampleIndices(int * const valid_sample_indices, const int * const invalid,
    const int samples, int valid_counts, cudaStream_t stream)
    */
    const size_t samples = flags->size;
    altar::cuda::bayesian::cudaMetropolis::setValidSampleIndices((int * const)indices->data,
        (const int * const)flags->data, samples, (int *)validCount->data);

    // all done
    // return
    Py_RETURN_NONE;
}

// queue valid samples
const char * const altar::cuda::extensions::cudaMetropolis::queueValidSamples__name__ = "cudaMetropolis_queueValidSamples";
const char * const altar::cuda::extensions::cudaMetropolis::queueValidSamples__doc__ =
    "cudaMetropolis label valid samples";

PyObject *
altar::cuda::extensions::cudaMetropolis::queueValidSamples(PyObject *, PyObject * args)
{
    PyObject * candidateCapsule, *proposalCapsule, *indicesCapsule;
    size_t validCount;

    int status = PyArg_ParseTuple(args, "O!O!O!k:cudaMetropolis_queueValidSamples",
                                    &PyCapsule_Type, &candidateCapsule,
                                    &PyCapsule_Type, &proposalCapsule,
                                    &PyCapsule_Type, &indicesCapsule,
                                    &validCount);
    if(!status) return 0;
    if (!PyCapsule_IsValid(indicesCapsule, altar::cuda::extensions::vector::capsule_t) ||
        !PyCapsule_IsValid(candidateCapsule, altar::cuda::extensions::matrix::capsule_t) ||
        !PyCapsule_IsValid(proposalCapsule, altar::cuda::extensions::matrix::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid matrix/vector capsule");
        return 0;
    }


    cuda_vector * indices = static_cast<cuda_vector *>
        (PyCapsule_GetPointer(indicesCapsule, altar::cuda::extensions::vector::capsule_t));
    cuda_matrix * candidate = static_cast<cuda_matrix *>
        (PyCapsule_GetPointer(candidateCapsule, altar::cuda::extensions::matrix::capsule_t));
    cuda_matrix * proposal = static_cast<cuda_matrix *>
        (PyCapsule_GetPointer(proposalCapsule, altar::cuda::extensions::matrix::capsule_t));


    /*
    template <typename realtype_t>
    void altar::cuda::bayesian::cudaMetropolis::
    queueValidSamples(realtype_t * const theta_candidate, const realtype_t * const theta_proposal,
        const int * const validSample_indices,
        const size_t valid_samples, const size_t parameters,
        cudaStream_t stream)
    */

    const size_t parameters = candidate->size2;
    switch(candidate->dtype) {
    case PYCUDA_FLOAT:
        altar::cuda::bayesian::cudaMetropolis::queueValidSamples<float>(
            (float *)candidate->data, (const float *)proposal->data,
            (const int *) indices->data, validCount, parameters);
        break;
    case PYCUDA_DOUBLE:
        altar::cuda::bayesian::cudaMetropolis::queueValidSamples<double>(
            (double *)candidate->data, (const double *)proposal->data,
            (const int *) indices->data, validCount, parameters);
        break;
    default:
        PyErr_SetString(PyExc_TypeError, "invalid datatype: only double/float are supported");
        return 0;
    }
    // all done
    // return none
    Py_RETURN_NONE;
}


// accept/reject procedure and update original state with accepted samples
const char * const altar::cuda::extensions::cudaMetropolis::metropolisUpdate__name__ = "cudaMetropolis_metropolisUpdate";
const char * const altar::cuda::extensions::cudaMetropolis::metropolisUpdate__doc__ =
    "cudaMetropolis accept/reject procedure and update original state with accepted samples";

PyObject *
altar::cuda::extensions::cudaMetropolis::metropolisUpdate(PyObject *, PyObject * args)
{
    PyObject *thetaCapsule, *priorCapsule, *dataCapsule, *posteriorCapsule;
    PyObject *cthetaCapsule, *cpriorCapsule, *cdataCapsule, *cposteriorCapsule;
    PyObject *diceCapsule, *acceptFlagsCapsule, *validIndicesCapsule;
    size_t validCount;

    int status = PyArg_ParseTuple(args, "O!O!O!O!O!O!O!O!O!O!O!k:cudaMetropolis_metropolisUpdate",
                                    &PyCapsule_Type, &thetaCapsule,
                                    &PyCapsule_Type, &priorCapsule,
                                    &PyCapsule_Type, &dataCapsule,
                                    &PyCapsule_Type, &posteriorCapsule,
                                    &PyCapsule_Type, &cthetaCapsule,
                                    &PyCapsule_Type, &cpriorCapsule,
                                    &PyCapsule_Type, &cdataCapsule,
                                    &PyCapsule_Type, &cposteriorCapsule,
                                    &PyCapsule_Type, &diceCapsule,
                                    &PyCapsule_Type, &acceptFlagsCapsule,
                                    &PyCapsule_Type, &validIndicesCapsule,
                                    &validCount);
    if(!status) return 0;
    // check the capsule types of input
    if (!PyCapsule_IsValid(thetaCapsule, altar::cuda::extensions::matrix::capsule_t) ||
        !PyCapsule_IsValid(priorCapsule, altar::cuda::extensions::vector::capsule_t) ||
        !PyCapsule_IsValid(dataCapsule, altar::cuda::extensions::vector::capsule_t) ||
        !PyCapsule_IsValid(posteriorCapsule, altar::cuda::extensions::vector::capsule_t) ||
        !PyCapsule_IsValid(cthetaCapsule, altar::cuda::extensions::matrix::capsule_t) ||
        !PyCapsule_IsValid(cpriorCapsule, altar::cuda::extensions::vector::capsule_t) ||
        !PyCapsule_IsValid(cdataCapsule, altar::cuda::extensions::vector::capsule_t) ||
        !PyCapsule_IsValid(cposteriorCapsule, altar::cuda::extensions::vector::capsule_t) ||
        !PyCapsule_IsValid(diceCapsule, altar::cuda::extensions::vector::capsule_t) ||
        !PyCapsule_IsValid(acceptFlagsCapsule, altar::cuda::extensions::vector::capsule_t) ||
        !PyCapsule_IsValid(validIndicesCapsule, altar::cuda::extensions::vector::capsule_t))
    {
        PyErr_SetString(PyExc_TypeError, "invalid matrix/vector capsule");
        return 0;
    }

    // cast capsules to c pointers
    cuda_matrix * theta = static_cast<cuda_matrix *>
        (PyCapsule_GetPointer(thetaCapsule, altar::cuda::extensions::matrix::capsule_t));
    cuda_vector * prior = static_cast<cuda_vector *>
        (PyCapsule_GetPointer(priorCapsule, altar::cuda::extensions::vector::capsule_t));
    cuda_vector * data = static_cast<cuda_vector *>
        (PyCapsule_GetPointer(dataCapsule, altar::cuda::extensions::vector::capsule_t));
    cuda_vector * posterior = static_cast<cuda_vector *>
        (PyCapsule_GetPointer(posteriorCapsule, altar::cuda::extensions::vector::capsule_t));
    cuda_matrix * ctheta = static_cast<cuda_matrix *>
        (PyCapsule_GetPointer(cthetaCapsule, altar::cuda::extensions::matrix::capsule_t));
    cuda_vector * cprior = static_cast<cuda_vector *>
        (PyCapsule_GetPointer(cpriorCapsule, altar::cuda::extensions::vector::capsule_t));
    cuda_vector * cdata = static_cast<cuda_vector *>
        (PyCapsule_GetPointer(cdataCapsule, altar::cuda::extensions::vector::capsule_t));
    cuda_vector * cposterior = static_cast<cuda_vector *>
        (PyCapsule_GetPointer(cposteriorCapsule, altar::cuda::extensions::vector::capsule_t));
    cuda_vector * dice = static_cast<cuda_vector *>
        (PyCapsule_GetPointer(diceCapsule, altar::cuda::extensions::vector::capsule_t));
    cuda_vector * acceptFlags = static_cast<cuda_vector *>
        (PyCapsule_GetPointer(acceptFlagsCapsule, altar::cuda::extensions::vector::capsule_t));
    cuda_vector * validIndices= static_cast<cuda_vector *>
        (PyCapsule_GetPointer(validIndicesCapsule, altar::cuda::extensions::vector::capsule_t));

    /*
    template <typename realtype_t>
    void altar::cuda::bayesian::cudaMetropolis::
    metropolisUpdate(realtype_t * const theta, realtype_t * const prior,
        realtype_t * const data, realtype_t * const posterior,
        const realtype_t * const theta_candidate, const realtype_t * const prior_candidate,
        const realtype_t * const data_candidate, const realtype_t * const posterior_candidate,
        const realtype_t * const dices, int * const acceptance_flag, const int * const valid_sample_indices,
        const int samples, const int parameters,
        cudaStream_t stream)
    */
    const size_t parameters = theta->size2;
    if(theta->dtype == PYCUDA_FLOAT)
        altar::cuda::bayesian::cudaMetropolis::metropolisUpdate<float>(
            (float *)theta->data, (float *)prior->data,
            (float *)data->data, (float *)posterior->data,
            (const float *)ctheta->data, (const float *)cprior->data,
            (const float *)cdata->data, (const float *)cposterior->data,
            (const float *)dice->data, (int *)acceptFlags->data, (const int *)validIndices->data,
            validCount, parameters);
    else
        altar::cuda::bayesian::cudaMetropolis::metropolisUpdate<double>(
            (double *)theta->data, (double *)prior->data,
            (double *)data->data, (double *)posterior->data,
            (const double *)ctheta->data, (const double *)cprior->data,
            (const double *)cdata->data, (const double *)cposterior->data,
            (const double *)dice->data, (int *)acceptFlags->data, (const int *)validIndices->data,
            validCount, parameters);

    // all done
    // return none
    Py_RETURN_NONE;
}

// end of file
