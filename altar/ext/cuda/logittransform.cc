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
#include "logittransform.h"

// the kernel launchers
#include <altar/cuda/distributions/cudaLogitTransform.h>


auto
altar::cuda::extensions::distributions::logittransform::__init__(py::module & distributions) -> void
{
    // theta[:, idx_begin:idx_end] <- low + (high-low)*sigmoid(theta), in place
    distributions.def(
        "cudaLogitTransform_tophysical",
        [](grid_t & theta, std::size_t idx_begin, std::size_t idx_end,
           double low, double high) -> void {
            auto format = theta.view().format;
            if (format.size() == 1 && format[0] == 'd') {
                altar::cuda::distributions::cudaLogitTransform::to_physical<double>(
                    regrid<double, 2>(theta), idx_begin, idx_end, low, high);
            } else if (format.size() == 1 && format[0] == 'f') {
                altar::cuda::distributions::cudaLogitTransform::to_physical<float>(
                    regrid<float, 2>(theta), idx_begin, idx_end,
                    static_cast<float>(low), static_cast<float>(high));
            } else {
                throw py::value_error("cudaLogitTransform_tophysical: unsupported grid cell type '" + format + "'");
            }
            cudaCheckError("cudaLogitTransform_tophysical");
            synchronize("cudaLogitTransform_tophysical");
        },
        "theta"_a, "idx_begin"_a, "idx_end"_a, "low"_a, "high"_a,
        "theta[:, idx_begin:idx_end] <- low + (high-low)*sigmoid(theta), in place");

    // the inverse of {cudaLogitTransform_tophysical}
    distributions.def(
        "cudaLogitTransform_tosampling",
        [](grid_t & theta, std::size_t idx_begin, std::size_t idx_end,
           double low, double high) -> void {
            auto format = theta.view().format;
            if (format.size() == 1 && format[0] == 'd') {
                altar::cuda::distributions::cudaLogitTransform::to_sampling<double>(
                    regrid<double, 2>(theta), idx_begin, idx_end, low, high);
            } else if (format.size() == 1 && format[0] == 'f') {
                altar::cuda::distributions::cudaLogitTransform::to_sampling<float>(
                    regrid<float, 2>(theta), idx_begin, idx_end,
                    static_cast<float>(low), static_cast<float>(high));
            } else {
                throw py::value_error("cudaLogitTransform_tosampling: unsupported grid cell type '" + format + "'");
            }
            cudaCheckError("cudaLogitTransform_tosampling");
            synchronize("cudaLogitTransform_tosampling");
        },
        "theta"_a, "idx_begin"_a, "idx_end"_a, "low"_a, "high"_a,
        "theta[:, idx_begin:idx_end] <- logit((theta-low)/(high-low)), in place");

    // fill {jacobian[:, idx_begin:idx_end]} with d(physical)/d(sampling); {theta} is
    // PHYSICAL space (read-only)
    distributions.def(
        "cudaLogitTransform_jacobian",
        [](grid_t & theta, grid_t & jacobian, std::size_t idx_begin, std::size_t idx_end,
           double low, double high) -> void {
            auto format = theta.view().format;
            if (format.size() == 1 && format[0] == 'd') {
                altar::cuda::distributions::cudaLogitTransform::jacobian<double>(
                    regrid<const double, 2>(theta), regrid<double, 2>(jacobian),
                    idx_begin, idx_end, low, high);
            } else if (format.size() == 1 && format[0] == 'f') {
                altar::cuda::distributions::cudaLogitTransform::jacobian<float>(
                    regrid<const float, 2>(theta), regrid<float, 2>(jacobian),
                    idx_begin, idx_end, static_cast<float>(low), static_cast<float>(high));
            } else {
                throw py::value_error("cudaLogitTransform_jacobian: unsupported grid cell type '" + format + "'");
            }
            cudaCheckError("cudaLogitTransform_jacobian");
            synchronize("cudaLogitTransform_jacobian");
        },
        "theta"_a, "jacobian"_a, "idx_begin"_a, "idx_end"_a, "low"_a, "high"_a,
        "jacobian[:, idx_begin:idx_end] <- (high-low)*sig*(1-sig), sig=(theta-low)/(high-low)");

    // fill {gradient[:, idx_begin:idx_end]} with d/d(sampling)[log(sig)+log(1-sig)]; {theta}
    // is PHYSICAL space (read-only)
    distributions.def(
        "cudaLogitTransform_jacobiangradient",
        [](grid_t & theta, grid_t & gradient, std::size_t idx_begin, std::size_t idx_end,
           double low, double high) -> void {
            auto format = theta.view().format;
            if (format.size() == 1 && format[0] == 'd') {
                altar::cuda::distributions::cudaLogitTransform::jacobian_gradient<double>(
                    regrid<const double, 2>(theta), regrid<double, 2>(gradient),
                    idx_begin, idx_end, low, high);
            } else if (format.size() == 1 && format[0] == 'f') {
                altar::cuda::distributions::cudaLogitTransform::jacobian_gradient<float>(
                    regrid<const float, 2>(theta), regrid<float, 2>(gradient),
                    idx_begin, idx_end, static_cast<float>(low), static_cast<float>(high));
            } else {
                throw py::value_error("cudaLogitTransform_jacobiangradient: unsupported grid cell type '" + format + "'");
            }
            cudaCheckError("cudaLogitTransform_jacobiangradient");
            synchronize("cudaLogitTransform_jacobiangradient");
        },
        "theta"_a, "gradient"_a, "idx_begin"_a, "idx_end"_a, "low"_a, "high"_a,
        "gradient[:, idx_begin:idx_end] <- 1 - 2*sig, sig=(theta-low)/(high-low)");

    // add each sample's standard-logistic log-pdf, summed over [idx_begin, idx_end), into
    // {likelihood} (samples,) -- an accumulation, matching {cudaUniform_logpdf}'s convention;
    // {theta} is PHYSICAL space (read-only)
    distributions.def(
        "cudaLogitTransform_logjacobian",
        [](grid_t & theta, grid_t & likelihood, std::size_t idx_begin, std::size_t idx_end,
           double low, double high) -> void {
            auto format = theta.view().format;
            if (format.size() == 1 && format[0] == 'd') {
                altar::cuda::distributions::cudaLogitTransform::log_jacobian<double>(
                    regrid<const double, 2>(theta), regrid<double, 1>(likelihood),
                    idx_begin, idx_end, low, high);
            } else if (format.size() == 1 && format[0] == 'f') {
                altar::cuda::distributions::cudaLogitTransform::log_jacobian<float>(
                    regrid<const float, 2>(theta), regrid<float, 1>(likelihood),
                    idx_begin, idx_end, static_cast<float>(low), static_cast<float>(high));
            } else {
                throw py::value_error("cudaLogitTransform_logjacobian: unsupported grid cell type '" + format + "'");
            }
            cudaCheckError("cudaLogitTransform_logjacobian");
            synchronize("cudaLogitTransform_logjacobian");
        },
        "theta"_a, "likelihood"_a, "idx_begin"_a, "idx_end"_a, "low"_a, "high"_a,
        "likelihood[s] += sum_{idx_begin..idx_end} log(sig) + log(1-sig)");

    // gradient[:, idx_begin:idx_end] <- gradient*J + (1 - 2*sig), in place; {theta} is
    // PHYSICAL space (read-only)
    distributions.def(
        "cudaLogitTransform_chaingradient",
        [](grid_t & theta, grid_t & gradient, std::size_t idx_begin, std::size_t idx_end,
           double low, double high) -> void {
            auto format = theta.view().format;
            if (format.size() == 1 && format[0] == 'd') {
                altar::cuda::distributions::cudaLogitTransform::chain_gradient<double>(
                    regrid<const double, 2>(theta), regrid<double, 2>(gradient),
                    idx_begin, idx_end, low, high);
            } else if (format.size() == 1 && format[0] == 'f') {
                altar::cuda::distributions::cudaLogitTransform::chain_gradient<float>(
                    regrid<const float, 2>(theta), regrid<float, 2>(gradient),
                    idx_begin, idx_end, static_cast<float>(low), static_cast<float>(high));
            } else {
                throw py::value_error("cudaLogitTransform_chaingradient: unsupported grid cell type '" + format + "'");
            }
            cudaCheckError("cudaLogitTransform_chaingradient");
            synchronize("cudaLogitTransform_chaingradient");
        },
        "theta"_a, "gradient"_a, "idx_begin"_a, "idx_end"_a, "low"_a, "high"_a,
        "gradient[:, idx_begin:idx_end] <- gradient*(high-low)*sig*(1-sig) + (1 - 2*sig)");
}


// end of file
