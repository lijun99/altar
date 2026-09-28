// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//

// my declarations
#include "moment.h"

// the kernel launchers
#include <altar/models/seismic/cuda/cudaMoment.h>


auto
altar::models::seismic::extensions::moment::__init__(py::module & m) -> void
{
    namespace kernels = altar::models::seismic::cudaMoment;

    // likelihood[s] += the moment constraint's log-density of sample {s}
    m.def(
        "cudaMoment_logpdf",
        [](grid_t & theta, grid_t & likelihood, std::size_t idx_begin, std::size_t idx_end,
           double mean, double sigma, grid_t & mu_area, double factor) -> void {
            auto format = theta.view().format;
            if (format.size() == 1 && format[0] == 'd') {
                kernels::logpdf<double>(
                    regrid<const double, 2>(theta), regrid<double, 1>(likelihood),
                    idx_begin, idx_end, mean, sigma, regrid<const double, 1>(mu_area), factor);
            } else if (format.size() == 1 && format[0] == 'f') {
                kernels::logpdf<float>(
                    regrid<const float, 2>(theta), regrid<float, 1>(likelihood),
                    idx_begin, idx_end, static_cast<float>(mean), static_cast<float>(sigma),
                    regrid<const float, 1>(mu_area), static_cast<float>(factor));
            } else {
                throw py::value_error("cudaMoment_logpdf: unsupported grid cell type '" + format + "'");
            }
            cudaCheckError("cudaMoment_logpdf");
            synchronize("cudaMoment_logpdf");
        },
        "theta"_a, "likelihood"_a, "idx_begin"_a, "idx_end"_a, "mean"_a, "sigma"_a,
        "mu_area"_a, "factor"_a,
        "likelihood[s] -= factor*(Mw - mean)^2/(2 sigma^2), Mw = (log10|sum_i mu_area[i] theta[s,i]| + 5.9)/1.5");

    // gradient[:, idx_begin:idx_end] <- d/dtheta of the moment constraint's log-density
    m.def(
        "cudaMoment_logpdfgradient",
        [](grid_t & theta, grid_t & gradient, std::size_t idx_begin, std::size_t idx_end,
           double mean, double sigma, grid_t & mu_area, double factor) -> void {
            auto format = theta.view().format;
            if (format.size() == 1 && format[0] == 'd') {
                kernels::logpdf_gradient<double>(
                    regrid<const double, 2>(theta), regrid<double, 2>(gradient),
                    idx_begin, idx_end, mean, sigma, regrid<const double, 1>(mu_area), factor);
            } else if (format.size() == 1 && format[0] == 'f') {
                kernels::logpdf_gradient<float>(
                    regrid<const float, 2>(theta), regrid<float, 2>(gradient),
                    idx_begin, idx_end, static_cast<float>(mean), static_cast<float>(sigma),
                    regrid<const float, 1>(mu_area), static_cast<float>(factor));
            } else {
                throw py::value_error("cudaMoment_logpdfgradient: unsupported grid cell type '" + format + "'");
            }
            cudaCheckError("cudaMoment_logpdfgradient");
            synchronize("cudaMoment_logpdfgradient");
        },
        "theta"_a, "gradient"_a, "idx_begin"_a, "idx_end"_a, "mean"_a, "sigma"_a,
        "mu_area"_a, "factor"_a,
        "gradient[:, idx_begin:idx_end] <- d/dtheta of the moment constraint's log-density");
}

// end of file
