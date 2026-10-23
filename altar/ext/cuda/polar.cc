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
#include "polar.h"

// the kernel launchers
#include <altar/cuda/distributions/cudaPolar.h>


auto
altar::cuda::extensions::distributions::polar::__init__(py::module & distributions) -> void
{
    // theta followed by the components of its slips sampled as a magnitude and a rake
    distributions.def(
        "cudaPolar_extend",
        [](grid_t & theta, grid_t & extended, std::size_t magnitude, std::size_t rake,
           std::size_t count) -> void {
            auto format = theta.view().format;
            if (format.size() == 1 && format[0] == 'd') {
                altar::cuda::distributions::cudaPolar::extend<double>(
                    regrid<const double, 2>(theta), regrid<double, 2>(extended), magnitude, rake, count);
            } else if (format.size() == 1 && format[0] == 'f') {
                altar::cuda::distributions::cudaPolar::extend<float>(
                    regrid<const float, 2>(theta), regrid<float, 2>(extended), magnitude, rake, count);
            } else {
                throw py::value_error("cudaPolar_extend: unsupported grid cell type '" + format + "'");
            }
            cudaCheckError("cudaPolar_extend");
            synchronize("cudaPolar_extend");
        },
        "theta"_a, "extended"_a, "magnitude"_a, "rake"_a, "count"_a,
        "extended = theta, then magnitude cos(rake), then magnitude sin(rake), the rake in degrees");

    // the chain rule of the extension, onto the columns of theta
    distributions.def(
        "cudaPolar_fold",
        [](grid_t & theta, grid_t & gradient_extended, grid_t & gradient, std::size_t magnitude,
           std::size_t rake, std::size_t count) -> void {
            auto format = theta.view().format;
            if (format.size() == 1 && format[0] == 'd') {
                altar::cuda::distributions::cudaPolar::fold<double>(
                    regrid<const double, 2>(theta), regrid<const double, 2>(gradient_extended),
                    regrid<double, 2>(gradient), magnitude, rake, count);
            } else if (format.size() == 1 && format[0] == 'f') {
                altar::cuda::distributions::cudaPolar::fold<float>(
                    regrid<const float, 2>(theta), regrid<const float, 2>(gradient_extended),
                    regrid<float, 2>(gradient), magnitude, rake, count);
            } else {
                throw py::value_error("cudaPolar_fold: unsupported grid cell type '" + format + "'");
            }
            cudaCheckError("cudaPolar_fold");
            synchronize("cudaPolar_fold");
        },
        "theta"_a, "gradient_extended"_a, "gradient"_a, "magnitude"_a, "rake"_a, "count"_a,
        "gradient = the gradient on the extended theta, folded onto theta's columns by the chain rule");
}


// end of file
