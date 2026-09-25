// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-2025 parasim inc
// (c) 2010-2025 california institute of technology
// all rights reserved

// externals
#include "external.h"
// my declarations
#include "logistic.h"

// the kernel launchers
#include <altar/cuda/distributions/cudaLogistic.h>


auto
altar::cuda::extensions::distributions::logistic::__init__(py::module & distributions) -> void
{
    // draw one sample per row of {theta}, for the parameters in [idx_begin, idx_end), from
    // the standard logistic distribution
    distributions.def(
        "cudaLogistic_sample",
        [](grid_t & theta, std::size_t idx_begin, std::size_t idx_end) -> void {
            auto format = theta.view().format;
            if (format.size() == 1 && format[0] == 'd') {
                altar::cuda::distributions::cudaLogistic::sample<double>(
                    regrid<double, 2>(theta), idx_begin, idx_end);
            } else if (format.size() == 1 && format[0] == 'f') {
                altar::cuda::distributions::cudaLogistic::sample<float>(
                    regrid<float, 2>(theta), idx_begin, idx_end);
            } else {
                throw py::value_error("cudaLogistic_sample: unsupported grid cell type '" + format + "'");
            }
            cudaCheckError("cudaLogistic_sample");
            synchronize("cudaLogistic_sample");
        },
        "theta"_a, "idx_begin"_a, "idx_end"_a,
        "draw theta[:, idx_begin:idx_end] from the standard logistic distribution, in place");

    // add each sample's log pdf, summed over [idx_begin, idx_end), into {probability}
    // (samples,) -- an accumulation, so the caller must zero (or otherwise pre-fill)
    // {probability} first if this is meant to be the only contribution
    distributions.def(
        "cudaLogistic_logpdf",
        [](grid_t & theta, grid_t & probability, std::size_t idx_begin, std::size_t idx_end) -> void {
            auto format = theta.view().format;
            if (format.size() == 1 && format[0] == 'd') {
                altar::cuda::distributions::cudaLogistic::logpdf<double>(
                    regrid<const double, 2>(theta), regrid<double, 1>(probability), idx_begin, idx_end);
            } else if (format.size() == 1 && format[0] == 'f') {
                altar::cuda::distributions::cudaLogistic::logpdf<float>(
                    regrid<const float, 2>(theta), regrid<float, 1>(probability), idx_begin, idx_end);
            } else {
                throw py::value_error("cudaLogistic_logpdf: unsupported grid cell type '" + format + "'");
            }
            cudaCheckError("cudaLogistic_logpdf");
            synchronize("cudaLogistic_logpdf");
        },
        "theta"_a, "probability"_a, "idx_begin"_a, "idx_end"_a,
        "probability[s] += sum(logpdf(theta[s, idx_begin:idx_end]))");

    // fill {probability} (samples x parameters) with the log pdf gradient with respect to
    // every parameter in [idx_begin, idx_end) -- a plain assignment, not an accumulation
    distributions.def(
        "cudaLogistic_logpdfgradient",
        [](grid_t & theta, grid_t & probability, std::size_t idx_begin, std::size_t idx_end) -> void {
            auto format = theta.view().format;
            if (format.size() == 1 && format[0] == 'd') {
                altar::cuda::distributions::cudaLogistic::logpdfgradient<double>(
                    regrid<const double, 2>(theta), regrid<double, 2>(probability), idx_begin, idx_end);
            } else if (format.size() == 1 && format[0] == 'f') {
                altar::cuda::distributions::cudaLogistic::logpdfgradient<float>(
                    regrid<const float, 2>(theta), regrid<float, 2>(probability), idx_begin, idx_end);
            } else {
                throw py::value_error("cudaLogistic_logpdfgradient: unsupported grid cell type '" + format + "'");
            }
            cudaCheckError("cudaLogistic_logpdfgradient");
            synchronize("cudaLogistic_logpdfgradient");
        },
        "theta"_a, "probability"_a, "idx_begin"_a, "idx_end"_a,
        "probability[s, idx_begin:idx_end] = d logpdf(theta[s, :]) / d theta[s, :]");
}


// end of file
