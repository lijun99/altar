// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//

// externals
#include "external.h"
// my declarations
#include "noise.h"

// the kernel launchers
#include <altar/cuda/distributions/cudaNoise.h>


auto
altar::cuda::extensions::distributions::noise::__init__(py::module & distributions) -> void
{
    // the data likelihoods of a model with its covariance scaled by exp(2h)
    distributions.def(
        "cudaNoise_likelihood",
        [](grid_t & theta, grid_t & likelihood, std::size_t column, double observations,
           double normalization, std::size_t batch) -> void {
            auto format = theta.view().format;
            if (format.size() == 1 && format[0] == 'd') {
                altar::cuda::distributions::cudaNoise::likelihood<double>(
                    regrid<const double, 2>(theta), regrid<double, 1>(likelihood), column,
                    observations, normalization, batch);
            } else if (format.size() == 1 && format[0] == 'f') {
                altar::cuda::distributions::cudaNoise::likelihood<float>(
                    regrid<const float, 2>(theta), regrid<float, 1>(likelihood), column,
                    static_cast<float>(observations), static_cast<float>(normalization), batch);
            } else {
                throw py::value_error("cudaNoise_likelihood: unsupported grid cell type '" + format + "'");
            }
            cudaCheckError("cudaNoise_likelihood");
            synchronize("cudaNoise_likelihood");
        },
        "theta"_a, "likelihood"_a, "column"_a, "observations"_a, "normalization"_a, "batch"_a,
        "likelihood = c - N h - exp(-2h) (c - likelihood), h in {column} of theta");

    // and their gradients
    distributions.def(
        "cudaNoise_gradient",
        [](grid_t & theta, grid_t & likelihood, grid_t & data_gradient, grid_t & gradient,
           std::size_t column, double observations, double normalization, std::size_t batch) -> void {
            auto format = theta.view().format;
            if (format.size() == 1 && format[0] == 'd') {
                altar::cuda::distributions::cudaNoise::gradient<double>(
                    regrid<const double, 2>(theta), regrid<const double, 1>(likelihood),
                    regrid<double, 2>(data_gradient), regrid<double, 2>(gradient), column,
                    observations, normalization, batch);
            } else if (format.size() == 1 && format[0] == 'f') {
                altar::cuda::distributions::cudaNoise::gradient<float>(
                    regrid<const float, 2>(theta), regrid<const float, 1>(likelihood),
                    regrid<float, 2>(data_gradient), regrid<float, 2>(gradient), column,
                    static_cast<float>(observations), static_cast<float>(normalization), batch);
            } else {
                throw py::value_error("cudaNoise_gradient: unsupported grid cell type '" + format + "'");
            }
            cudaCheckError("cudaNoise_gradient");
            synchronize("cudaNoise_gradient");
        },
        "theta"_a, "likelihood"_a, "data_gradient"_a, "gradient"_a, "column"_a, "observations"_a,
        "normalization"_a, "batch"_a,
        "data_gradient *= exp(-2h); gradient[:, column] += -N + 2 exp(-2h) (c - likelihood), "
        "likelihood the unscaled one");
}


// end of file
