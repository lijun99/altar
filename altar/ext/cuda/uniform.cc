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
#include "uniform.h"

// the kernel launchers
#include <altar/cuda/distributions/cudaUniform.h>


auto
altar::cuda::extensions::distributions::uniform::__init__(py::module & distributions) -> void
{
    // draw one sample per row of {theta}, for the parameters in [idx_begin, idx_end),
    // uniform over [low, high) -- the same bounds for every parameter in range
    distributions.def(
        "cudaUniform_sample",
        [](grid_t & theta, std::size_t idx_begin, std::size_t idx_end,
           double low, double high) -> void {
            auto format = theta.view().format;
            if (format.size() == 1 && format[0] == 'd') {
                altar::cuda::distributions::cudaUniform::sample<double>(
                    regrid<double, 2>(theta), idx_begin, idx_end, low, high);
            } else if (format.size() == 1 && format[0] == 'f') {
                altar::cuda::distributions::cudaUniform::sample<float>(
                    regrid<float, 2>(theta), idx_begin, idx_end,
                    static_cast<float>(low), static_cast<float>(high));
            } else {
                throw py::value_error("cudaUniform_sample: unsupported grid cell type '" + format + "'");
            }
            cudaCheckError("cudaUniform_sample");
            synchronize("cudaUniform_sample");
        },
        "theta"_a, "idx_begin"_a, "idx_end"_a, "low"_a, "high"_a,
        "draw theta[:, idx_begin:idx_end] ~ U[low, high), in place");

    // the per-parameter-bounds counterpart of {cudaUniform_sample}: {low[j]}/{high[j]} apply
    // to parameter {idx_begin + j}
    distributions.def(
        "cudaUniform_sample_unique",
        [](grid_t & theta, std::size_t idx_begin, std::size_t idx_end,
           grid_t & low, grid_t & high) -> void {
            auto format = theta.view().format;
            if (format.size() == 1 && format[0] == 'd') {
                altar::cuda::distributions::cudaUniform::sample_unique<double>(
                    regrid<double, 2>(theta), idx_begin, idx_end,
                    regrid<const double, 1>(low), regrid<const double, 1>(high));
            } else if (format.size() == 1 && format[0] == 'f') {
                altar::cuda::distributions::cudaUniform::sample_unique<float>(
                    regrid<float, 2>(theta), idx_begin, idx_end,
                    regrid<const float, 1>(low), regrid<const float, 1>(high));
            } else {
                throw py::value_error("cudaUniform_sample_unique: unsupported grid cell type '" + format + "'");
            }
            cudaCheckError("cudaUniform_sample_unique");
            synchronize("cudaUniform_sample_unique");
        },
        "theta"_a, "idx_begin"_a, "idx_end"_a, "low"_a, "high"_a,
        "draw theta[:, idx_begin+j] ~ U[low[j], high[j]), in place");

    // add each sample's log pdf, summed over [idx_begin, idx_end), into {probability}
    // (samples,) -- an accumulation, so the caller must zero (or otherwise pre-fill)
    // {probability} first if this is meant to be the only contribution
    distributions.def(
        "cudaUniform_logpdf",
        [](grid_t & theta, grid_t & probability, std::size_t idx_begin, std::size_t idx_end,
           double low, double high) -> void {
            auto format = theta.view().format;
            if (format.size() == 1 && format[0] == 'd') {
                altar::cuda::distributions::cudaUniform::logpdf<double>(
                    regrid<const double, 2>(theta), regrid<double, 1>(probability),
                    idx_begin, idx_end, low, high);
            } else if (format.size() == 1 && format[0] == 'f') {
                altar::cuda::distributions::cudaUniform::logpdf<float>(
                    regrid<const float, 2>(theta), regrid<float, 1>(probability),
                    idx_begin, idx_end, static_cast<float>(low), static_cast<float>(high));
            } else {
                throw py::value_error("cudaUniform_logpdf: unsupported grid cell type '" + format + "'");
            }
            cudaCheckError("cudaUniform_logpdf");
            synchronize("cudaUniform_logpdf");
        },
        "theta"_a, "probability"_a, "idx_begin"_a, "idx_end"_a, "low"_a, "high"_a,
        "probability[s] += (idx_end - idx_begin) * log(1 / (high - low))");

    // the per-parameter-bounds counterpart of {cudaUniform_logpdf}
    distributions.def(
        "cudaUniform_logpdf_unique",
        [](grid_t & theta, grid_t & probability, std::size_t idx_begin, std::size_t idx_end,
           grid_t & low, grid_t & high) -> void {
            auto format = theta.view().format;
            if (format.size() == 1 && format[0] == 'd') {
                altar::cuda::distributions::cudaUniform::logpdf_unique<double>(
                    regrid<const double, 2>(theta), regrid<double, 1>(probability),
                    idx_begin, idx_end, regrid<const double, 1>(low), regrid<const double, 1>(high));
            } else if (format.size() == 1 && format[0] == 'f') {
                altar::cuda::distributions::cudaUniform::logpdf_unique<float>(
                    regrid<const float, 2>(theta), regrid<float, 1>(probability),
                    idx_begin, idx_end, regrid<const float, 1>(low), regrid<const float, 1>(high));
            } else {
                throw py::value_error("cudaUniform_logpdf_unique: unsupported grid cell type '" + format + "'");
            }
            cudaCheckError("cudaUniform_logpdf_unique");
            synchronize("cudaUniform_logpdf_unique");
        },
        "theta"_a, "probability"_a, "idx_begin"_a, "idx_end"_a, "low"_a, "high"_a,
        "probability[s] += sum_j log(1 / (high[j] - low[j]))");

    // the logistic-edged uniform: accumulates into {probability}, like {cudaUniform_logpdf}
    distributions.def(
        "cudaUniform_softlogpdf",
        [](grid_t & theta, grid_t & probability, std::size_t idx_begin, std::size_t idx_end,
           double low, double high, double sharpness) -> void {
            auto format = theta.view().format;
            if (format.size() == 1 && format[0] == 'd') {
                altar::cuda::distributions::cudaUniform::soft_logpdf<double>(
                    regrid<const double, 2>(theta), regrid<double, 1>(probability),
                    idx_begin, idx_end, low, high, sharpness);
            } else if (format.size() == 1 && format[0] == 'f') {
                altar::cuda::distributions::cudaUniform::soft_logpdf<float>(
                    regrid<const float, 2>(theta), regrid<float, 1>(probability),
                    idx_begin, idx_end, static_cast<float>(low), static_cast<float>(high),
                    static_cast<float>(sharpness));
            } else {
                throw py::value_error("cudaUniform_softlogpdf: unsupported grid cell type '" + format + "'");
            }
            cudaCheckError("cudaUniform_softlogpdf");
            synchronize("cudaUniform_softlogpdf");
        },
        "theta"_a, "probability"_a, "idx_begin"_a, "idx_end"_a, "low"_a, "high"_a, "sharpness"_a,
        "probability[s] += the log pdf of the logistic-edged uniform, summed over my parameters");

    // and the gradient of its log pdf
    distributions.def(
        "cudaUniform_softgradient",
        [](grid_t & theta, grid_t & gradient, std::size_t idx_begin, std::size_t idx_end,
           double low, double high, double sharpness) -> void {
            auto format = theta.view().format;
            if (format.size() == 1 && format[0] == 'd') {
                altar::cuda::distributions::cudaUniform::soft_gradient<double>(
                    regrid<const double, 2>(theta), regrid<double, 2>(gradient),
                    idx_begin, idx_end, low, high, sharpness);
            } else if (format.size() == 1 && format[0] == 'f') {
                altar::cuda::distributions::cudaUniform::soft_gradient<float>(
                    regrid<const float, 2>(theta), regrid<float, 2>(gradient),
                    idx_begin, idx_end, static_cast<float>(low), static_cast<float>(high),
                    static_cast<float>(sharpness));
            } else {
                throw py::value_error("cudaUniform_softgradient: unsupported grid cell type '" + format + "'");
            }
            cudaCheckError("cudaUniform_softgradient");
            synchronize("cudaUniform_softgradient");
        },
        "theta"_a, "gradient"_a, "idx_begin"_a, "idx_end"_a, "low"_a, "high"_a, "sharpness"_a,
        "gradient[:, idx_begin:idx_end] <- the gradient of the logistic-edged uniform's log pdf");
}


// end of file
