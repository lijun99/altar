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


// declarations
#include "kinematicg.h"

// c++ class includes
#include <altar/models/seismic/cuda/cudaKinematicG.h>

// local includes
#include "capsules.h"

// cuda utilities
#include <pyre/cuda.h>
#include <pyre/cuda/cudaext.h>


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
const char * const altar::extensions::models::cudaseismic::kinematicg_alloc__name__ = "kinematicg_alloc";
const char * const altar::extensions::models::cudaseismic::kinematicg_alloc__doc__ = "allocate kinematicg model";

PyObject *
altar::extensions::models::cudaseismic::
kinematicg_alloc(PyObject *, PyObject * args)
{
    // parameters
    size_t Nas, Ndd, Nmesh;
    double dsp; 
    size_t Nt, Npt;
    double dt;
    PyObject * gt0sCap;
    size_t samples, parameters, observations;
    PyObject * gidxmapCap;
    int dtype;
    // we don't accept any arguments, so check that we didn't get any
    int status = PyArg_ParseTuple(args,
                                "kkkdkkdO!kkkO!i:kinematicg_alloc",
                                &Nas, &Ndd, &Nmesh, &dsp, 
                                &Nt, &Npt, &dt,
                                &PyCapsule_Type, &gt0sCap,
                                &samples, &parameters, &observations, 
                                &PyCapsule_Type, &gidxmapCap, &dtype);
    // if something went wrong
    if (!status) {
        // complain
        return 0;
    }
    
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(gt0sCap, vector::capsule_t) ||
            !PyCapsule_IsValid(gidxmapCap, vector::capsule_t)) 
    {
        PyErr_SetString(PyExc_TypeError, "invalid capsule for kinematicg_alloc");
        return 0;
    }

    // convert PyObjects to C Objects
    cuda_vector * gt0s = static_cast<cuda_vector *>
        (PyCapsule_GetPointer(gt0sCap, vector::capsule_t));
    cuda_vector * gidxMap = static_cast<cuda_vector *>
        (PyCapsule_GetPointer(gidxmapCap, vector::capsule_t));    
        
    // make a cuda/c kinematicg model
    if (dtype == PYCUDA_FLOAT) {
        SModel_t * cmodel = new SModel_t(
            Nas, Ndd, Nmesh, dsp, 
            Nt, Npt, dt, 
            (const float * const)gt0s->data,
            samples, parameters, observations, 
            (const size_t * const)gidxMap->data);
        return PyCapsule_New(cmodel, kgSmodel_capsule, kinematicg_free);
    }
    else if ( dtype == PYCUDA_DOUBLE) {
        DModel_t * cmodel = new DModel_t(
            Nas, Ndd, Nmesh, dsp, 
            Nt, Npt, dt, 
            (const double * const)gt0s->data,
            samples, parameters, observations, 
            (const size_t * const)gidxMap->data);
        return PyCapsule_New(cmodel, kgDmodel_capsule, kinematicg_free);
    }
    else {
        return NULL; //error 
    }
}

void 
altar::extensions::models::cudaseismic::
kinematicg_free(PyObject * capsule)
{
    // bail out if the capsule is not valid
    if (PyCapsule_IsValid(capsule, kgSmodel_capsule)) {
        SModel_t * cmodel = static_cast<SModel_t *>(PyCapsule_GetPointer(capsule, kgSmodel_capsule));
        delete cmodel;
    }
    else if (PyCapsule_IsValid(capsule, kgDmodel_capsule)) {
        DModel_t * cmodel = static_cast<DModel_t *>(PyCapsule_GetPointer(capsule, kgDmodel_capsule));
        delete cmodel;
    }
    // and return
    return;
}

// forward model
const char * const altar::extensions::models::cudaseismic::kinematicg_forward__name__ = "kinematicg_forward";
const char * const altar::extensions::models::cudaseismic::kinematicg_forward__doc__ = "forwardate kinematicg model";

PyObject *
altar::extensions::models::cudaseismic::
kinematicg_forward(PyObject *, PyObject * args)
{
    //inputs
    PyObject * modelCap, *handleCap;
    PyObject * thetaCap, * GbCap, *DpredCap; 
    size_t batch, parameters;
    int return_residual;
    
    int status = PyArg_ParseTuple(args,
                            "O!O!O!O!O!kkp:kinematicg_forward",
                            &PyCapsule_Type, &handleCap,
                            &PyCapsule_Type, &modelCap,
                            &PyCapsule_Type, &thetaCap,
                            &PyCapsule_Type, &GbCap,
                            &PyCapsule_Type, &DpredCap,
                            &parameters, &batch, &return_residual);
    if (!status) {
        // complain
        return 0;
    }
    
    
    if (!PyCapsule_IsValid(thetaCap, matrix::capsule_t) || !PyCapsule_IsValid(GbCap, matrix::capsule_t)
        || !PyCapsule_IsValid(DpredCap, matrix::capsule_t))
    {
        PyErr_SetString(PyExc_TypeError, "invalid input matrix capsule for kinematicg_forward");
        return 0;
    }

    cublasHandle_t handle = static_cast<cublasHandle_t>(PyCapsule_GetPointer(handleCap, pyre::extensions::cuda::cublas::capsule_t));
    
    // get c objects from capsules
    cuda_matrix * theta = static_cast<cuda_matrix *> (PyCapsule_GetPointer(thetaCap, matrix::capsule_t));
    cuda_matrix * Gb = static_cast<cuda_matrix *> (PyCapsule_GetPointer(GbCap, matrix::capsule_t));
    cuda_matrix * dpred = static_cast<cuda_matrix *> (PyCapsule_GetPointer(DpredCap, matrix::capsule_t));

        // bail out if the capsule is not valid
    if (PyCapsule_IsValid(modelCap, kgSmodel_capsule)) 
    {
        SModel_t * cmodel = static_cast<SModel_t *>(PyCapsule_GetPointer(modelCap, kgSmodel_capsule));
        cmodel->forwardModel(handle, (const float * const)theta->data, (const float * const)Gb->data,
                    (float * const)dpred->data, parameters, batch, return_residual);
    }    
    else if (PyCapsule_IsValid(modelCap, kgDmodel_capsule)) 
    {
        DModel_t * cmodel = static_cast<DModel_t *>(PyCapsule_GetPointer(modelCap, kgDmodel_capsule));
        cmodel->forwardModel(handle, (const double * const)theta->data, (const double * const)Gb->data,
                    (double * const)dpred->data, parameters, batch, return_residual);
    }   
    else {
        PyErr_SetString(PyExc_TypeError, "invalid model capsule for kinematicg_forward");
        return 0;
    }
    
    // return none
    Py_RETURN_NONE;
}
// end of file
