// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-2025 parasim inc
// (c) 2010-2025 california institute of technology
// all rights reserved
//
// Author(s): Codex

#include <portinfo>
#include <Python.h>

#include <altar/cuda/bayesian/cudaLeapfrog.h>

#include <pyre/cuda.h>
#include <pyre/cuda/capsules.h>

#include "capsules.h"
#include "leapfrog.h"

namespace altar { namespace cuda { namespace extensions { namespace cudaLeapfrog {

namespace {
    inline bool checkMatrix(PyObject * capsule) {
        return PyCapsule_IsValid(capsule, altar::cuda::extensions::matrix::capsule_t);
    }

    inline bool checkVector(PyObject * capsule) {
        return PyCapsule_IsValid(capsule, altar::cuda::extensions::vector::capsule_t);
    }
}

const char * const sampleMomentum__name__ = "cudaHMC_sample_momentum";
const char * const sampleMomentum__doc__ = "Sample standard normal momenta on the GPU.";
PyObject * sampleMomentum(PyObject *, PyObject * args) {
    PyObject * momentumCapsule;
    unsigned long samples, parameters;
    if (!PyArg_ParseTuple(args, "O!kk:cudaHMC_sample_momentum",
                          &PyCapsule_Type, &momentumCapsule,
                          &samples, &parameters)) {
        return nullptr;
    }
    if (!checkMatrix(momentumCapsule)) {
        PyErr_SetString(PyExc_TypeError, "cudaHMC_sample_momentum invalid matrix capsule");
        return nullptr;
    }
    auto * momentum = static_cast<cuda_matrix *>(
        PyCapsule_GetPointer(momentumCapsule, altar::cuda::extensions::matrix::capsule_t));
    switch (momentum->dtype) {
    case PYCUDA_FLOAT:
        altar::cuda::bayesian::cudaLeapfrog::sampleMomentum<float>(
            reinterpret_cast<float *>(momentum->data), momentum->size1, momentum->size2);
        break;
    case PYCUDA_DOUBLE:
        altar::cuda::bayesian::cudaLeapfrog::sampleMomentum<double>(
            reinterpret_cast<double *>(momentum->data), momentum->size1, momentum->size2);
        break;
    default:
        PyErr_SetString(PyExc_TypeError, "cudaHMC_sample_momentum only supports float/double");
        return nullptr;
    }
    Py_RETURN_NONE;
}

const char * const computePotentialAndGradient__name__ = "cudaHMC_compute_potential_and_gradient";
const char * const computePotentialAndGradient__doc__ =
    "Compute potential energy and gradients for HMC.";
PyObject * computePotentialAndGradient(PyObject *, PyObject * args) {
    PyObject * priorCapsule, * dataCapsule, * gradPriorCapsule, * gradDataCapsule;
    PyObject * potentialCapsule, * gradCapsule, * jacobianCapsule = Py_None;
    unsigned long samples, parameters;
    double beta;
    int reparam;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!Okkdi:cudaHMC_compute_potential_and_gradient",
                          &PyCapsule_Type, &priorCapsule,
                          &PyCapsule_Type, &dataCapsule,
                          &PyCapsule_Type, &gradPriorCapsule,
                          &PyCapsule_Type, &gradDataCapsule,
                          &PyCapsule_Type, &potentialCapsule,
                          &PyCapsule_Type, &gradCapsule,
                          &jacobianCapsule,
                          &samples, &parameters, &beta, &reparam)) {
        return nullptr;
    }
    if (!checkVector(priorCapsule) || !checkVector(dataCapsule) ||
        !checkMatrix(gradPriorCapsule) || !checkMatrix(gradDataCapsule) ||
        !checkVector(potentialCapsule) || !checkMatrix(gradCapsule)) {
        PyErr_SetString(PyExc_TypeError, "cudaHMC_compute_potential_and_gradient invalid capsules");
        return nullptr;
    }
    auto * prior = static_cast<cuda_vector *>(PyCapsule_GetPointer(priorCapsule, altar::cuda::extensions::vector::capsule_t));
    auto * data = static_cast<cuda_vector *>(PyCapsule_GetPointer(dataCapsule, altar::cuda::extensions::vector::capsule_t));
    auto * gradPrior = static_cast<cuda_matrix *>(PyCapsule_GetPointer(gradPriorCapsule, altar::cuda::extensions::matrix::capsule_t));
    auto * gradData = static_cast<cuda_matrix *>(PyCapsule_GetPointer(gradDataCapsule, altar::cuda::extensions::matrix::capsule_t));
    auto * potential = static_cast<cuda_vector *>(PyCapsule_GetPointer(potentialCapsule, altar::cuda::extensions::vector::capsule_t));
    auto * gradPotential = static_cast<cuda_matrix *>(PyCapsule_GetPointer(gradCapsule, altar::cuda::extensions::matrix::capsule_t));

    const bool hasJacobian = (jacobianCapsule != Py_None) && checkMatrix(jacobianCapsule);
    auto * jacobian = hasJacobian
        ? static_cast<cuda_matrix *>(PyCapsule_GetPointer(jacobianCapsule, altar::cuda::extensions::matrix::capsule_t))
        : nullptr;

    switch (potential->dtype) {
    case PYCUDA_FLOAT:
        altar::cuda::bayesian::cudaLeapfrog::computePotentialAndGradient<float>(
            reinterpret_cast<float *>(prior->data), reinterpret_cast<float *>(data->data),
            reinterpret_cast<float *>(gradPrior->data), reinterpret_cast<float *>(gradData->data),
            reinterpret_cast<float *>(potential->data), reinterpret_cast<float *>(gradPotential->data),
            jacobian ? reinterpret_cast<float *>(jacobian->data) : nullptr,
            samples, parameters, static_cast<float>(beta), reparam != 0);
        break;
    case PYCUDA_DOUBLE:
        altar::cuda::bayesian::cudaLeapfrog::computePotentialAndGradient<double>(
            reinterpret_cast<double *>(prior->data), reinterpret_cast<double *>(data->data),
            reinterpret_cast<double *>(gradPrior->data), reinterpret_cast<double *>(gradData->data),
            reinterpret_cast<double *>(potential->data), reinterpret_cast<double *>(gradPotential->data),
            jacobian ? reinterpret_cast<double *>(jacobian->data) : nullptr,
            samples, parameters, beta, reparam != 0);
        break;
    default:
        PyErr_SetString(PyExc_TypeError, "cudaHMC_compute_potential_and_gradient only supports float/double");
        return nullptr;
    }
    Py_RETURN_NONE;
}

const char * const kineticEnergy__name__ = "cudaHMC_kinetic_energy";
const char * const kineticEnergy__doc__ = "Compute kinetic energies for each chain.";
PyObject * kineticEnergy(PyObject *, PyObject * args) {
    PyObject * momentumCapsule, * kineticCapsule;
    unsigned long samples, parameters;
    if (!PyArg_ParseTuple(args, "O!O!kk:cudaHMC_kinetic_energy",
                          &PyCapsule_Type, &momentumCapsule,
                          &PyCapsule_Type, &kineticCapsule,
                          &samples, &parameters)) {
        return nullptr;
    }
    if (!checkMatrix(momentumCapsule) || !checkVector(kineticCapsule)) {
        PyErr_SetString(PyExc_TypeError, "cudaHMC_kinetic_energy invalid capsules");
        return nullptr;
    }
    auto * momentum = static_cast<cuda_matrix *>(PyCapsule_GetPointer(momentumCapsule, altar::cuda::extensions::matrix::capsule_t));
    auto * kinetic = static_cast<cuda_vector *>(PyCapsule_GetPointer(kineticCapsule, altar::cuda::extensions::vector::capsule_t));
    switch (kinetic->dtype) {
    case PYCUDA_FLOAT:
        altar::cuda::bayesian::cudaLeapfrog::kineticEnergy<float>(
            reinterpret_cast<float *>(momentum->data), reinterpret_cast<float *>(kinetic->data),
            samples, parameters);
        break;
    case PYCUDA_DOUBLE:
        altar::cuda::bayesian::cudaLeapfrog::kineticEnergy<double>(
            reinterpret_cast<double *>(momentum->data), reinterpret_cast<double *>(kinetic->data),
            samples, parameters);
        break;
    default:
        PyErr_SetString(PyExc_TypeError, "cudaHMC_kinetic_energy only supports float/double");
        return nullptr;
    }
    Py_RETURN_NONE;
}

const char * const updatePosition__name__ = "cudaHMC_update_position";
const char * const updatePosition__doc__ = "theta += step * momentum";
PyObject * updatePosition(PyObject *, PyObject * args) {
    PyObject * thetaCapsule, * momentumCapsule;
    unsigned long samples, parameters;
    double step;
    if (!PyArg_ParseTuple(args, "O!O!kkd:cudaHMC_update_position",
                          &PyCapsule_Type, &thetaCapsule,
                          &PyCapsule_Type, &momentumCapsule,
                          &samples, &parameters, &step)) {
        return nullptr;
    }
    if (!checkMatrix(thetaCapsule) || !checkMatrix(momentumCapsule)) {
        PyErr_SetString(PyExc_TypeError, "cudaHMC_update_position invalid matrix capsule");
        return nullptr;
    }
    auto * theta = static_cast<cuda_matrix *>(PyCapsule_GetPointer(thetaCapsule, altar::cuda::extensions::matrix::capsule_t));
    auto * momentum = static_cast<cuda_matrix *>(PyCapsule_GetPointer(momentumCapsule, altar::cuda::extensions::matrix::capsule_t));
    switch (theta->dtype) {
    case PYCUDA_FLOAT:
        altar::cuda::bayesian::cudaLeapfrog::updatePosition<float>(
            reinterpret_cast<float *>(theta->data), reinterpret_cast<float *>(momentum->data),
            samples, parameters, static_cast<float>(step));
        break;
    case PYCUDA_DOUBLE:
        altar::cuda::bayesian::cudaLeapfrog::updatePosition<double>(
            reinterpret_cast<double *>(theta->data), reinterpret_cast<double *>(momentum->data),
            samples, parameters, step);
        break;
    default:
        PyErr_SetString(PyExc_TypeError, "cudaHMC_update_position only supports float/double");
        return nullptr;
    }
    Py_RETURN_NONE;
}

const char * const updateMomentum__name__ = "cudaHMC_update_momentum";
const char * const updateMomentum__doc__ = "momentum += scale * gradU";
PyObject * updateMomentum(PyObject *, PyObject * args) {
    PyObject * momentumCapsule, * gradCapsule;
    unsigned long samples, parameters;
    double scale;
    if (!PyArg_ParseTuple(args, "O!O!kkd:cudaHMC_update_momentum",
                          &PyCapsule_Type, &momentumCapsule,
                          &PyCapsule_Type, &gradCapsule,
                          &samples, &parameters, &scale)) {
        return nullptr;
    }
    if (!checkMatrix(momentumCapsule) || !checkMatrix(gradCapsule)) {
        PyErr_SetString(PyExc_TypeError, "cudaHMC_update_momentum invalid matrix capsule");
        return nullptr;
    }
    auto * momentum = static_cast<cuda_matrix *>(PyCapsule_GetPointer(momentumCapsule, altar::cuda::extensions::matrix::capsule_t));
    auto * grad = static_cast<cuda_matrix *>(PyCapsule_GetPointer(gradCapsule, altar::cuda::extensions::matrix::capsule_t));
    switch (momentum->dtype) {
    case PYCUDA_FLOAT:
        altar::cuda::bayesian::cudaLeapfrog::updateMomentum<float>(
            reinterpret_cast<float *>(momentum->data), reinterpret_cast<float *>(grad->data),
            samples, parameters, static_cast<float>(scale));
        break;
    case PYCUDA_DOUBLE:
        altar::cuda::bayesian::cudaLeapfrog::updateMomentum<double>(
            reinterpret_cast<double *>(momentum->data), reinterpret_cast<double *>(grad->data),
            samples, parameters, scale);
        break;
    default:
        PyErr_SetString(PyExc_TypeError, "cudaHMC_update_momentum only supports float/double");
        return nullptr;
    }
    Py_RETURN_NONE;
}

const char * const metropolis__name__ = "cudaHMC_metropolis_hastings";
const char * const metropolis__doc__ = "Metropolis-Hastings accept/reject step on the GPU.";
PyObject * metropolis(PyObject *, PyObject * args) {
    PyObject * deltaHCapsule, * maskCapsule;
    unsigned long samples;
    if (!PyArg_ParseTuple(args, "O!O!k:cudaHMC_metropolis_hastings",
                          &PyCapsule_Type, &deltaHCapsule,
                          &PyCapsule_Type, &maskCapsule,
                          &samples)) {
        return nullptr;
    }
    if (!checkVector(deltaHCapsule) || !checkVector(maskCapsule)) {
        PyErr_SetString(PyExc_TypeError, "cudaHMC_metropolis_hastings invalid capsules");
        return nullptr;
    }
    auto * deltaH = static_cast<cuda_vector *>(PyCapsule_GetPointer(deltaHCapsule, altar::cuda::extensions::vector::capsule_t));
    auto * mask = static_cast<cuda_vector *>(PyCapsule_GetPointer(maskCapsule, altar::cuda::extensions::vector::capsule_t));
    switch (deltaH->dtype) {
    case PYCUDA_FLOAT:
        altar::cuda::bayesian::cudaLeapfrog::metropolis<float>(
            reinterpret_cast<float *>(deltaH->data), reinterpret_cast<int *>(mask->data), samples);
        break;
    case PYCUDA_DOUBLE:
        altar::cuda::bayesian::cudaLeapfrog::metropolis<double>(
            reinterpret_cast<double *>(deltaH->data), reinterpret_cast<int *>(mask->data), samples);
        break;
    default:
        PyErr_SetString(PyExc_TypeError, "cudaHMC_metropolis_hastings only supports float/double delta_H");
        return nullptr;
    }
    Py_RETURN_NONE;
}

const char * const restoreRejected__name__ = "cudaHMC_restore_rejected";
const char * const restoreRejected__doc__ = "Restore theta/momentum for rejected proposals.";
PyObject * restoreRejected(PyObject *, PyObject * args) {
    PyObject * thetaCapsule, * thetaOldCapsule, * momentumCapsule, * momentumOldCapsule;
    PyObject * maskCapsule;
    unsigned long samples, parameters;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!kk:cudaHMC_restore_rejected",
                          &PyCapsule_Type, &thetaCapsule,
                          &PyCapsule_Type, &thetaOldCapsule,
                          &PyCapsule_Type, &momentumCapsule,
                          &PyCapsule_Type, &momentumOldCapsule,
                          &PyCapsule_Type, &maskCapsule,
                          &samples, &parameters)) {
        return nullptr;
    }
    if (!checkMatrix(thetaCapsule) || !checkMatrix(thetaOldCapsule) ||
        !checkMatrix(momentumCapsule) || !checkMatrix(momentumOldCapsule) ||
        !checkVector(maskCapsule)) {
        PyErr_SetString(PyExc_TypeError, "cudaHMC_restore_rejected invalid capsules");
        return nullptr;
    }
    auto * theta = static_cast<cuda_matrix *>(PyCapsule_GetPointer(thetaCapsule, altar::cuda::extensions::matrix::capsule_t));
    auto * thetaOld = static_cast<cuda_matrix *>(PyCapsule_GetPointer(thetaOldCapsule, altar::cuda::extensions::matrix::capsule_t));
    auto * momentum = static_cast<cuda_matrix *>(PyCapsule_GetPointer(momentumCapsule, altar::cuda::extensions::matrix::capsule_t));
    auto * momentumOld = static_cast<cuda_matrix *>(PyCapsule_GetPointer(momentumOldCapsule, altar::cuda::extensions::matrix::capsule_t));
    auto * mask = static_cast<cuda_vector *>(PyCapsule_GetPointer(maskCapsule, altar::cuda::extensions::vector::capsule_t));
    switch (theta->dtype) {
    case PYCUDA_FLOAT:
        altar::cuda::bayesian::cudaLeapfrog::restoreRejected<float>(
            reinterpret_cast<float *>(theta->data), reinterpret_cast<float *>(thetaOld->data),
            reinterpret_cast<float *>(momentum->data), reinterpret_cast<float *>(momentumOld->data),
            reinterpret_cast<int *>(mask->data), samples, parameters);
        break;
    case PYCUDA_DOUBLE:
        altar::cuda::bayesian::cudaLeapfrog::restoreRejected<double>(
            reinterpret_cast<double *>(theta->data), reinterpret_cast<double *>(thetaOld->data),
            reinterpret_cast<double *>(momentum->data), reinterpret_cast<double *>(momentumOld->data),
            reinterpret_cast<int *>(mask->data), samples, parameters);
        break;
    default:
        PyErr_SetString(PyExc_TypeError, "cudaHMC_restore_rejected only supports float/double");
        return nullptr;
    }
    Py_RETURN_NONE;
}

const char * const restoreMatrix__name__ = "cudaHMC_restore_rejected_matrix";
const char * const restoreMatrix__doc__ = "Restore auxiliary matrices for rejected proposals.";
PyObject * restoreMatrix(PyObject *, PyObject * args) {
    PyObject * currentCapsule, * backupCapsule, * maskCapsule;
    unsigned long samples, parameters;
    if (!PyArg_ParseTuple(args, "O!O!O!kk:cudaHMC_restore_rejected_matrix",
                          &PyCapsule_Type, &currentCapsule,
                          &PyCapsule_Type, &backupCapsule,
                          &PyCapsule_Type, &maskCapsule,
                          &samples, &parameters)) {
        return nullptr;
    }
    if (!checkMatrix(currentCapsule) || !checkMatrix(backupCapsule) || !checkVector(maskCapsule)) {
        PyErr_SetString(PyExc_TypeError, "cudaHMC_restore_rejected_matrix invalid capsules");
        return nullptr;
    }
    auto * current = static_cast<cuda_matrix *>(PyCapsule_GetPointer(currentCapsule, altar::cuda::extensions::matrix::capsule_t));
    auto * backup = static_cast<cuda_matrix *>(PyCapsule_GetPointer(backupCapsule, altar::cuda::extensions::matrix::capsule_t));
    auto * mask = static_cast<cuda_vector *>(PyCapsule_GetPointer(maskCapsule, altar::cuda::extensions::vector::capsule_t));
    switch (current->dtype) {
    case PYCUDA_FLOAT:
        altar::cuda::bayesian::cudaLeapfrog::restoreMatrix<float>(
            reinterpret_cast<float *>(current->data), reinterpret_cast<float *>(backup->data),
            reinterpret_cast<int *>(mask->data), samples, parameters);
        break;
    case PYCUDA_DOUBLE:
        altar::cuda::bayesian::cudaLeapfrog::restoreMatrix<double>(
            reinterpret_cast<double *>(current->data), reinterpret_cast<double *>(backup->data),
            reinterpret_cast<int *>(mask->data), samples, parameters);
        break;
    default:
        PyErr_SetString(PyExc_TypeError, "cudaHMC_restore_rejected_matrix only supports float/double");
        return nullptr;
    }
    Py_RETURN_NONE;
}

}}}} // namespace altar::cuda::extensions::cudaLeapfrog

// end of file
