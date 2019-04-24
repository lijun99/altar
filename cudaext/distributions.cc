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
#include "distributions.h"

// c++ class includes
#include <altar/cuda/distributions/cudaUniform.h>
#include <altar/cuda/distributions/cudaGaussian.h>

// local includes
#include "capsules.h"

// cuda utilities
#include <pyre/cuda.h>
#include <pyre/cuda/cudaext.h>

// cudaUniform distribution 

// cudaUniform_sample
const char * const altar::cuda::extensions::cudaUniform::sample__name__ = "cudaUniform_sample";
const char * const altar::cuda::extensions::cudaUniform::sample__doc__ = "cudaUniform rng for a matrix";

PyObject *
altar::cuda::extensions::cudaUniform::sample(PyObject *, PyObject * args) {
    // the arguments
    // sample(theta, samples, (idx_begin, idx_end), (low, high))

    PyObject * thetaCapsule;
    size_t idx_begin, idx_end; // parameter index
    double low, high; // support or range
    size_t samples;
    
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!k(kk)(dd):cudaUniform_sample",
                                  &PyCapsule_Type, &thetaCapsule, &samples,
                                  &idx_begin, &idx_end, &low, &high
                                  );
    // if something went wrong
    if (!status) return 0;
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(thetaCapsule, altar::cuda::extensions::matrix::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid capsule for cudaUniform_sample");
        return 0;
    }
    
    // convert PyObjects to C Objects
    cuda_matrix * theta = static_cast<cuda_matrix *>
        (PyCapsule_GetPointer(thetaCapsule, altar::cuda::extensions::matrix::capsule_t));
        
    size_t parameters = theta->size2;

    // call c method
    if(theta->dtype == PYCUDA_FLOAT) //single precision
    {
        altar::cuda::distributions::cudaUniform::sample<float>
            ((float *)theta->data, samples, parameters, idx_begin, idx_end, (float)low, (float)high);
    }
    else //double precision 
    {
        altar::cuda::distributions::cudaUniform::sample<double>
            ((double *)theta->data, samples, parameters, idx_begin, idx_end, low, high);
    }

    // all done
    // return None                                                                                                                             
    Py_INCREF(Py_None);
    return Py_None;

}

const char * const altar::cuda::extensions::cudaUniform::logpdf__name__ = "cudaUniform_logpdf";
const char * const altar::cuda::extensions::cudaUniform::logpdf__doc__ =
    "cudaUniform compute log pdf";

PyObject *
altar::cuda::extensions::cudaUniform::logpdf(PyObject *, PyObject * args) {
    // the arguments
    // sample(theta, probability, samples, (idx_begin, idx_end), (low, high))

    PyObject * thetaCapsule, * probabilityCapsule;
    size_t idx_begin, idx_end; // parameter index
    double low, high; // support or range
    size_t samples;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!O!k(kk)(dd):cudaUniform_logpdf",
                                  &PyCapsule_Type, &thetaCapsule,
                                  &PyCapsule_Type, &probabilityCapsule, 
                                  &samples, &idx_begin, &idx_end,
                                  &low, &high
                                  );
        // if something went wrong
    if (!status) return 0;
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(thetaCapsule, altar::cuda::extensions::matrix::capsule_t) 
            || !PyCapsule_IsValid(probabilityCapsule, altar::cuda::extensions::vector::capsule_t) ) 
    {
        PyErr_SetString(PyExc_TypeError, "invalid capsule for cudaUniform_logpdf");
        return 0;
    }
    
    // convert PyObjects to C Objects
    cuda_matrix * theta = static_cast<cuda_matrix *>
        (PyCapsule_GetPointer(thetaCapsule, altar::cuda::extensions::matrix::capsule_t));
    cuda_vector * prob = static_cast<cuda_vector *>
        (PyCapsule_GetPointer(probabilityCapsule, altar::cuda::extensions::vector::capsule_t));
        
    size_t parameters = theta->size2;

    // call c method
    if(theta->dtype == PYCUDA_FLOAT) //single precision
    {
        altar::cuda::distributions::cudaUniform::logpdf<float>
            ((const float *)theta->data, (float *)prob->data,
            samples, parameters, idx_begin, idx_end, (float)low, (float)high);
    }
    else //double precision 
    {
        altar::cuda::distributions::cudaUniform::logpdf<double>
            ((const double *)theta->data, (double *)prob->data,
            samples, parameters, idx_begin, idx_end, low, high);
    }
    // all done
    // return None                                                                                                                             
    Py_INCREF(Py_None);
    return Py_None;
}

// cudaUniform_verify
const char * const altar::cuda::extensions::cudaUniform::verify__name__ = "cudaUniform_verify";
const char * const altar::cuda::extensions::cudaUniform::verify__doc__ =
    "cudaUniform verify range";

PyObject *
altar::cuda::extensions::cudaUniform::verify(PyObject *, PyObject * args) {
    // the arguments
    // verify(theta, flag, samples, (idx_begin, idx_end), (low, high))

    PyObject * thetaCapsule, * flagCapsule;
    size_t idx_begin, idx_end; // parameter index
    double low, high; // support or range
    size_t samples;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!O!k(kk)(dd):cudaUniform_verify",
                                  &PyCapsule_Type, &thetaCapsule,
                                  &PyCapsule_Type, &flagCapsule, 
                                  &samples, &idx_begin, &idx_end,
                                  &low, &high
                                  );
    // if something went wrong
    if (!status) return 0;
    if (!PyCapsule_IsValid(thetaCapsule, altar::cuda::extensions::matrix::capsule_t) 
            || !PyCapsule_IsValid(flagCapsule, altar::cuda::extensions::vector::capsule_t) ) 
    {
        PyErr_SetString(PyExc_TypeError, "invalid capsule for cudaUniform_verify");
        return 0;
    }
    
    // convert PyObjects to C Objects
    cuda_matrix * theta = static_cast<cuda_matrix *>
        (PyCapsule_GetPointer(thetaCapsule, altar::cuda::extensions::matrix::capsule_t));
    cuda_vector * flag = static_cast<cuda_vector *>
        (PyCapsule_GetPointer(flagCapsule, altar::cuda::extensions::vector::capsule_t));
        
    size_t parameters = theta->size2;

    // call c method
    if(theta->dtype == PYCUDA_FLOAT) //single precision
    {
        altar::cuda::distributions::cudaUniform::verify<float>
            ((float *)theta->data, (int *)flag->data,
            samples, parameters, idx_begin, idx_end, (float)low, (float)high);
    }
    else //double precision 
    {
        altar::cuda::distributions::cudaUniform::verify<double>
            ((double *)theta->data, (int *)flag->data,
            samples, parameters, idx_begin, idx_end, low, high);
    }
    // all done
    // return None                                                                                                                             
    Py_INCREF(Py_None);
    return Py_None;
}


// cudaGaussian distribution 

// cudaGaussian_sample
const char * const altar::cuda::extensions::cudaGaussian::sample__name__ = "cudaGaussian_sample";
const char * const altar::cuda::extensions::cudaGaussian::sample__doc__ = "cudaGaussian rng for a matrix";

PyObject *
altar::cuda::extensions::cudaGaussian::sample(PyObject *, PyObject * args) {
    // the arguments
    // sample(theta, samples, (idx_begin, idx_end), (mean, sigma))

    PyObject * thetaCapsule;
    size_t idx_begin, idx_end; // parameter index
    double mean, sigma; // support or range
    size_t samples;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!k(kk)(dd):cudaGaussian_sample",
                                  &PyCapsule_Type, &thetaCapsule, &samples,
                                  &idx_begin, &idx_end, &mean, &sigma
                                  );
    // if something went wrong
    if (!status) return 0;
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(thetaCapsule, altar::cuda::extensions::matrix::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid capsule for cudaGaussian_sample");
        return 0;
    }
    
    // convert PyObjects to C Objects
    cuda_matrix * theta = static_cast<cuda_matrix *>
        (PyCapsule_GetPointer(thetaCapsule, altar::cuda::extensions::matrix::capsule_t));
        
    size_t parameters = theta->size2;

    // call c method
    switch(theta->dtype) {
    case PYCUDA_FLOAT: //single precision
        altar::cuda::distributions::cudaGaussian::sample<float>
            ((float *)theta->data, samples, parameters, idx_begin, idx_end, (float)mean, (float)sigma);
        break;
    case PYCUDA_DOUBLE: //double precision 
        altar::cuda::distributions::cudaGaussian::sample<double>
            ((double *)theta->data, samples, parameters, idx_begin, idx_end, mean, sigma);
        break;
    default:
        PyErr_SetString(PyExc_TypeError, "invalid data type for cudaGaussian_sample");
        return 0;
    }
    // all done
    // return None                                                                                                                             
    Py_INCREF(Py_None);
    return Py_None;

}

const char * const altar::cuda::extensions::cudaGaussian::logpdf__name__ = "cudaGaussian_logpdf";
const char * const altar::cuda::extensions::cudaGaussian::logpdf__doc__ =
    "cudaGaussian compute log pdf";

PyObject *
altar::cuda::extensions::cudaGaussian::logpdf(PyObject *, PyObject * args) {
    // the arguments
    // sample(theta, probability, samples, (idx_begin, idx_end), (mean, sigma))

    PyObject * thetaCapsule, * probabilityCapsule;
    size_t idx_begin, idx_end; // parameter index
    double mean, sigma; // support or range
    size_t samples;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!O!k(kk)(dd):cudaGaussian_logpdf",
                                  &PyCapsule_Type, &thetaCapsule,
                                  &PyCapsule_Type, &probabilityCapsule, 
                                  &samples, &idx_begin, &idx_end,
                                  &mean, &sigma
                                  );
        // if something went wrong
    if (!status) return 0;
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(thetaCapsule, altar::cuda::extensions::matrix::capsule_t) 
            || !PyCapsule_IsValid(probabilityCapsule, altar::cuda::extensions::vector::capsule_t) ) 
    {
        PyErr_SetString(PyExc_TypeError, "invalid capsule for cudaGaussian_logpdf");
        return 0;
    }
    
    // convert PyObjects to C Objects
    cuda_matrix * theta = static_cast<cuda_matrix *>
        (PyCapsule_GetPointer(thetaCapsule, altar::cuda::extensions::matrix::capsule_t));
    cuda_vector * prob = static_cast<cuda_vector *>
        (PyCapsule_GetPointer(probabilityCapsule, altar::cuda::extensions::vector::capsule_t));
        
    size_t parameters = theta->size2;

    // call c method
    /*  template <typename real_type>
        void altar::cuda::distributions::cudaGaussian::
        logpdf(const real_type * const theta, real_type * const probability, 
                    const size_t samples, const size_t parameters, 
                    const size_t idx_begin, const size_t idx_end,
                    const real_type mean, const real_type sigma,
                    cudaStream_t stream)
    */
    if(theta->dtype == PYCUDA_FLOAT) //single precision
    {
        altar::cuda::distributions::cudaGaussian::logpdf<float>
            ((const float *)theta->data, (float *)prob->data,
            samples, parameters, idx_begin, idx_end, (float)mean, (float)sigma);
    }
    else //double precision 
    {
        altar::cuda::distributions::cudaGaussian::logpdf<double>
            ((const double *)theta->data, (double *)prob->data,
            samples, parameters, idx_begin, idx_end, mean, sigma);
    }
    // all done
    // return None                                                                                                                             
    Py_INCREF(Py_None);
    return Py_None;
}

// end of file
