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
#include "gaussian.h"

// the kernel launchers
#include <altar/cuda/distributions/cudaGaussian.h>


auto
altar::cuda::extensions::distributions::gaussian::__init__(py::module & distributions) -> void
{
    // draw one sample per row of {theta}, for the parameters in [idx_begin, idx_end); {theta}
    // is (samples x parameters), written in place
    distributions.def(
        "cudaGaussian_sample",
        [](grid_t & theta, std::size_t idx_begin, std::size_t idx_end,
           double mean, double sigma) -> void {
            auto format = theta.view().format;
            if (format.size() == 1 && format[0] == 'd') {
                altar::cuda::distributions::cudaGaussian::sample<double>(
                    regrid<double, 2>(theta), idx_begin, idx_end, mean, sigma);
            } else if (format.size() == 1 && format[0] == 'f') {
                altar::cuda::distributions::cudaGaussian::sample<float>(
                    regrid<float, 2>(theta), idx_begin, idx_end,
                    static_cast<float>(mean), static_cast<float>(sigma));
            } else {
                throw py::value_error("cudaGaussian_sample: unsupported grid cell type '" + format + "'");
            }
            cudaCheckError("cudaGaussian_sample");
            synchronize("cudaGaussian_sample");
        },
        "theta"_a, "idx_begin"_a, "idx_end"_a, "mean"_a, "sigma"_a,
        "draw theta[:, idx_begin:idx_end] ~ N(mean, sigma^2), in place");

    // add each sample's log pdf, summed over [idx_begin, idx_end), into {probability}
    // (samples,) -- an accumulation, so the caller must zero (or otherwise pre-fill)
    // {probability} first if this is meant to be the only contribution
    distributions.def(
        "cudaGaussian_logpdf",
        [](grid_t & theta, grid_t & probability, std::size_t idx_begin, std::size_t idx_end,
           double mean, double sigma) -> void {
            auto format = theta.view().format;
            if (format.size() == 1 && format[0] == 'd') {
                altar::cuda::distributions::cudaGaussian::logpdf<double>(
                    regrid<const double, 2>(theta), regrid<double, 1>(probability),
                    idx_begin, idx_end, mean, sigma);
            } else if (format.size() == 1 && format[0] == 'f') {
                altar::cuda::distributions::cudaGaussian::logpdf<float>(
                    regrid<const float, 2>(theta), regrid<float, 1>(probability),
                    idx_begin, idx_end, static_cast<float>(mean), static_cast<float>(sigma));
            } else {
                throw py::value_error("cudaGaussian_logpdf: unsupported grid cell type '" + format + "'");
            }
            cudaCheckError("cudaGaussian_logpdf");
            synchronize("cudaGaussian_logpdf");
        },
        "theta"_a, "probability"_a, "idx_begin"_a, "idx_end"_a, "mean"_a, "sigma"_a,
        "probability[s] += sum(logpdf(theta[s, idx_begin:idx_end]))");

    // add the log pdf gradient with respect to parameter {index} into {probability}
    // (samples,); a no-op for samples where {index} falls outside [idx_begin, idx_end)
    distributions.def(
        "cudaGaussian_logpdfgradient_i",
        [](grid_t & theta, grid_t & probability, std::size_t idx_begin, std::size_t idx_end,
           std::size_t index, double mean, double sigma) -> void {
            auto format = theta.view().format;
            if (format.size() == 1 && format[0] == 'd') {
                altar::cuda::distributions::cudaGaussian::logpdfgradient_i<double>(
                    regrid<const double, 2>(theta), regrid<double, 1>(probability),
                    idx_begin, idx_end, index, mean, sigma);
            } else if (format.size() == 1 && format[0] == 'f') {
                altar::cuda::distributions::cudaGaussian::logpdfgradient_i<float>(
                    regrid<const float, 2>(theta), regrid<float, 1>(probability),
                    idx_begin, idx_end, index, static_cast<float>(mean), static_cast<float>(sigma));
            } else {
                throw py::value_error("cudaGaussian_logpdfgradient_i: unsupported grid cell type '" + format + "'");
            }
            cudaCheckError("cudaGaussian_logpdfgradient_i");
            synchronize("cudaGaussian_logpdfgradient_i");
        },
        "theta"_a, "probability"_a, "idx_begin"_a, "idx_end"_a, "index"_a, "mean"_a, "sigma"_a,
        "probability[s] += d logpdf(theta[s, index]) / d theta[s, index]");

    // fill {probability} (samples x parameters) with the log pdf gradient with respect to
    // every parameter in [idx_begin, idx_end) -- a plain assignment, not an accumulation,
    // unlike {cudaGaussian_logpdfgradient_i} above
    distributions.def(
        "cudaGaussian_logpdfgradient",
        [](grid_t & theta, grid_t & probability, std::size_t idx_begin, std::size_t idx_end,
           double mean, double sigma) -> void {
            auto format = theta.view().format;
            if (format.size() == 1 && format[0] == 'd') {
                altar::cuda::distributions::cudaGaussian::logpdfgradient<double>(
                    regrid<const double, 2>(theta), regrid<double, 2>(probability),
                    idx_begin, idx_end, mean, sigma);
            } else if (format.size() == 1 && format[0] == 'f') {
                altar::cuda::distributions::cudaGaussian::logpdfgradient<float>(
                    regrid<const float, 2>(theta), regrid<float, 2>(probability),
                    idx_begin, idx_end, static_cast<float>(mean), static_cast<float>(sigma));
            } else {
                throw py::value_error("cudaGaussian_logpdfgradient: unsupported grid cell type '" + format + "'");
            }
            cudaCheckError("cudaGaussian_logpdfgradient");
            synchronize("cudaGaussian_logpdfgradient");
        },
        "theta"_a, "probability"_a, "idx_begin"_a, "idx_end"_a, "mean"_a, "sigma"_a,
        "probability[s, idx_begin:idx_end] = d logpdf(theta[s, :]) / d theta[s, :]");
}


// end of file
