// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-2025 parasim inc
// (c) 2010-2025 california institute of technology
// all rights reserved

// externals
#include "external.h"
// my declarations
#include "tgaussian.h"

// the kernel launchers
#include <altar/cuda/distributions/cudaTGaussian.h>


auto
altar::cuda::extensions::distributions::tgaussian::__init__(py::module & distributions) -> void
{
    // draw one sample per row of {theta}, for the parameters in [idx_begin, idx_end); {low}/
    // {high} are the *normalized* (Phi-space) support bounds
    distributions.def(
        "cudaTGaussian_sample",
        [](grid_t & theta, std::size_t idx_begin, std::size_t idx_end,
           double mean, double sigma, double low, double high) -> void {
            auto format = theta.view().format;
            if (format.size() == 1 && format[0] == 'd') {
                altar::cuda::distributions::cudaTGaussian::sample<double>(
                    regrid<double, 2>(theta), idx_begin, idx_end, mean, sigma, low, high);
            } else if (format.size() == 1 && format[0] == 'f') {
                altar::cuda::distributions::cudaTGaussian::sample<float>(
                    regrid<float, 2>(theta), idx_begin, idx_end,
                    static_cast<float>(mean), static_cast<float>(sigma),
                    static_cast<float>(low), static_cast<float>(high));
            } else {
                throw py::value_error("cudaTGaussian_sample: unsupported grid cell type '" + format + "'");
            }
            cudaCheckError("cudaTGaussian_sample");
            synchronize("cudaTGaussian_sample");
        },
        "theta"_a, "idx_begin"_a, "idx_end"_a, "mean"_a, "sigma"_a, "low"_a, "high"_a,
        "draw theta[:, idx_begin:idx_end] from N(mean, sigma^2) truncated to the normalized "
        "support [low, high), in place");

    // add each sample's log pdf, summed over [idx_begin, idx_end), into {probability}
    // (samples,) -- an accumulation, so the caller must zero (or otherwise pre-fill)
    // {probability} first if this is meant to be the only contribution
    distributions.def(
        "cudaTGaussian_logpdf",
        [](grid_t & theta, grid_t & probability, std::size_t idx_begin, std::size_t idx_end,
           double mean, double sigma, double low, double high) -> void {
            auto format = theta.view().format;
            if (format.size() == 1 && format[0] == 'd') {
                altar::cuda::distributions::cudaTGaussian::logpdf<double>(
                    regrid<const double, 2>(theta), regrid<double, 1>(probability),
                    idx_begin, idx_end, mean, sigma, low, high);
            } else if (format.size() == 1 && format[0] == 'f') {
                altar::cuda::distributions::cudaTGaussian::logpdf<float>(
                    regrid<const float, 2>(theta), regrid<float, 1>(probability),
                    idx_begin, idx_end, static_cast<float>(mean), static_cast<float>(sigma),
                    static_cast<float>(low), static_cast<float>(high));
            } else {
                throw py::value_error("cudaTGaussian_logpdf: unsupported grid cell type '" + format + "'");
            }
            cudaCheckError("cudaTGaussian_logpdf");
            synchronize("cudaTGaussian_logpdf");
        },
        "theta"_a, "probability"_a, "idx_begin"_a, "idx_end"_a, "mean"_a, "sigma"_a, "low"_a, "high"_a,
        "probability[s] += sum(logpdf(theta[s, idx_begin:idx_end])), truncated to [low, high)");
}


// end of file
