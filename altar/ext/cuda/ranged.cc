// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-2025 parasim inc
// (c) 2010-2025 california institute of technology
// all rights reserved

// externals
#include "external.h"
// my declarations
#include "ranged.h"

// the kernel launchers
#include <altar/cuda/distributions/cudaRanged.h>


auto
altar::cuda::extensions::distributions::ranged::__init__(py::module & distributions) -> void
{
    // flag each sample whose parameters in [idx_begin, idx_end) fall outside [low, high] by
    // setting {invalid[sample] = 1}; a sample already flagged is left alone
    distributions.def(
        "cudaRanged_verify",
        [](grid_t & theta, grid_t & invalid, std::size_t idx_begin, std::size_t idx_end,
           double low, double high) -> void {
            auto format = theta.view().format;
            if (format.size() == 1 && format[0] == 'd') {
                altar::cuda::distributions::cudaRanged::verify<double>(
                    regrid<const double, 2>(theta), regrid<int, 1>(invalid), idx_begin, idx_end, low, high);
            } else if (format.size() == 1 && format[0] == 'f') {
                altar::cuda::distributions::cudaRanged::verify<float>(
                    regrid<const float, 2>(theta), regrid<int, 1>(invalid), idx_begin, idx_end,
                    static_cast<float>(low), static_cast<float>(high));
            } else {
                throw py::value_error("cudaRanged_verify: unsupported grid cell type '" + format + "'");
            }
            cudaCheckError("cudaRanged_verify");
            synchronize("cudaRanged_verify");
        },
        "theta"_a, "invalid"_a, "idx_begin"_a, "idx_end"_a, "low"_a, "high"_a,
        "invalid[s] = 1 if any of theta[s, idx_begin:idx_end] falls outside [low, high]");

    // the per-parameter-bounds counterpart of {cudaRanged_verify}
    distributions.def(
        "cudaRanged_verify_unique",
        [](grid_t & theta, grid_t & invalid, std::size_t idx_begin, std::size_t idx_end,
           grid_t & low, grid_t & high) -> void {
            auto format = theta.view().format;
            if (format.size() == 1 && format[0] == 'd') {
                altar::cuda::distributions::cudaRanged::verify_unique<double>(
                    regrid<const double, 2>(theta), regrid<int, 1>(invalid), idx_begin, idx_end,
                    regrid<const double, 1>(low), regrid<const double, 1>(high));
            } else if (format.size() == 1 && format[0] == 'f') {
                altar::cuda::distributions::cudaRanged::verify_unique<float>(
                    regrid<const float, 2>(theta), regrid<int, 1>(invalid), idx_begin, idx_end,
                    regrid<const float, 1>(low), regrid<const float, 1>(high));
            } else {
                throw py::value_error("cudaRanged_verify_unique: unsupported grid cell type '" + format + "'");
            }
            cudaCheckError("cudaRanged_verify_unique");
            synchronize("cudaRanged_verify_unique");
        },
        "theta"_a, "invalid"_a, "idx_begin"_a, "idx_end"_a, "low"_a, "high"_a,
        "invalid[s] = 1 if any of theta[s, idx_begin+j] falls outside [low[j], high[j]]");

    // clamp {theta[:, idx_begin:idx_end]} into [low, high], in place
    distributions.def(
        "cudaRanged_constrain",
        [](grid_t & theta, std::size_t idx_begin, std::size_t idx_end,
           double low, double high) -> void {
            auto format = theta.view().format;
            if (format.size() == 1 && format[0] == 'd') {
                altar::cuda::distributions::cudaRanged::constrain<double>(
                    regrid<double, 2>(theta), idx_begin, idx_end, low, high);
            } else if (format.size() == 1 && format[0] == 'f') {
                altar::cuda::distributions::cudaRanged::constrain<float>(
                    regrid<float, 2>(theta), idx_begin, idx_end,
                    static_cast<float>(low), static_cast<float>(high));
            } else {
                throw py::value_error("cudaRanged_constrain: unsupported grid cell type '" + format + "'");
            }
            cudaCheckError("cudaRanged_constrain");
            synchronize("cudaRanged_constrain");
        },
        "theta"_a, "idx_begin"_a, "idx_end"_a, "low"_a, "high"_a,
        "clamp theta[:, idx_begin:idx_end] into [low, high], in place");
}


// end of file
