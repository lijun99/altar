// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-2026 parasim inc
// (c) 2010-2026 california institute of technology
// all rights reserved
//
// Author(s): AlTar-1 team, rearranged by Lijun Zhu

// external dependencies
#include "external.h"
// namespace setup
#include "forward.h"

#include <gsl/gsl_blas.h>
#include <gsl/gsl_eigen.h>


void
altar::py::condition(py::module & m)
{
    m.def(
        "matrix_condition",
        [](gsl_matrix & sigma, double eval_ratio_min) -> void {
            // build my debugging channel
            pyre::journal::debug_t debug("altar.matrix_condition");

            // get matrix size
            size_t n = sigma.size1;

            // solve the eigen value problem
            gsl_vector * eval = gsl_vector_alloc(n);
            gsl_matrix * evec = gsl_matrix_alloc(n, n);
            gsl_eigen_symmv_workspace * w = gsl_eigen_symmv_alloc(n);

            gsl_eigen_symmv(&sigma, eval, evec, w);
            gsl_eigen_symmv_free(w);

            // sort the eigen values in ascending order (magnitude)
            gsl_eigen_symmv_sort(eval, evec, GSL_EIGEN_SORT_ABS_ASC);

            // make a transpose of the eigen vector matrix
            gsl_matrix * evecT = gsl_matrix_alloc(n, n);
            gsl_matrix_transpose_memcpy(evecT, evec);

            // allocate a matrix for conditioned eigen values
            gsl_matrix * diagM = gsl_matrix_calloc(n, n);

            // set the minimum eigen value as the max * ratio
            double eval_min = eval_ratio_min * gsl_vector_get(eval, n - 1);
            // copy the eigenvalues, set it to eval_min if smaller
            for (size_t i = 0; i < n; i++) {
                double eval_i = gsl_vector_get(eval, i);
                if (eval_i < eval_min) {
                    gsl_matrix_set(diagM, i, i, eval_min);
                } else {
                    gsl_matrix_set(diagM, i, i, eval_i);
                }
            }

            // reconstruct sigma from the conditioned eigen values
            gsl_matrix * tmp = gsl_matrix_alloc(n, n);
            gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, diagM, evecT, 0.0, tmp);
            gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, evec, tmp, 0.0, &sigma);

            // make sigma symmetric
            gsl_matrix_transpose_memcpy(tmp, &sigma);
            gsl_matrix_add(&sigma, tmp);
            gsl_matrix_scale(&sigma, 0.5);

            // free temporary data
            gsl_vector_free(eval);
            gsl_matrix_free(evec);
            gsl_matrix_free(evecT);
            gsl_matrix_free(diagM);
            gsl_matrix_free(tmp);

            // all done
            return;
        },
        "sigma"_a, "eval_ratio_min"_a,
        "condition a matrix to be positive definite: replace negative or small "
        "eigenvalues with ratio*max_eigenvalue");

    return;
}

// end of file
