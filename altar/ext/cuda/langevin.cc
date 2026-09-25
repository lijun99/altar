// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-2025 parasim inc
// (c) 2010-2025 california institute of technology
// all rights reserved

// externals
#include "external.h"
// my declarations
#include "langevin.h"

// the kernel launchers
#include <altar/cuda/bayesian/cudaLangevin.h>


auto
altar::cuda::extensions::langevin::__init__(py::module & m) -> void
{
    auto langevin = m.def_submodule("langevin", "stochastic gradient langevin dynamics (SGLD)");

    // the SGLD update, one parameter at a time:
    // theta[:, index] += half_epsilon_t * (prior_gradient + datalikelihood_gradient) + eta_t
    langevin.def(
        "cudaLangevin_updateTheta",
        [](grid_t & theta, grid_t & prior_gradient, grid_t & datalikelihood_gradient,
           double half_epsilon_t, grid_t & eta_t, std::size_t index) -> void {
            auto format = theta.view().format;
            if (format.size() == 1 && format[0] == 'd') {
                altar::cuda::bayesian::cudaLangevin::updateTheta<double>(
                    regrid<double, 2>(theta), regrid<const double, 1>(prior_gradient),
                    regrid<const double, 1>(datalikelihood_gradient), half_epsilon_t,
                    regrid<const double, 1>(eta_t), index);
            } else if (format.size() == 1 && format[0] == 'f') {
                altar::cuda::bayesian::cudaLangevin::updateTheta<float>(
                    regrid<float, 2>(theta), regrid<const float, 1>(prior_gradient),
                    regrid<const float, 1>(datalikelihood_gradient), static_cast<float>(half_epsilon_t),
                    regrid<const float, 1>(eta_t), index);
            } else {
                throw py::value_error("cudaLangevin_updateTheta: unsupported grid cell type '" + format + "'");
            }
            cudaCheckError("cudaLangevin_updateTheta");
            synchronize("cudaLangevin_updateTheta");
        },
        "theta"_a, "prior_gradient"_a, "datalikelihood_gradient"_a, "half_epsilon_t"_a, "eta_t"_a, "index"_a,
        "theta[:, index] += half_epsilon_t * (prior_gradient + datalikelihood_gradient) + eta_t");

    // the batched SGLD update, every parameter at once:
    // theta += half_epsilon_t * (alpha1*prior_gradient + alpha2*datalikelihood_gradient) + eta_t
    langevin.def(
        "cudaLangevin_updateThetaBatched",
        [](grid_t & theta, double alpha1, grid_t & prior_gradient,
           double alpha2, grid_t & datalikelihood_gradient,
           double half_epsilon_t, grid_t & eta_t) -> void {
            auto format = theta.view().format;
            if (format.size() == 1 && format[0] == 'd') {
                altar::cuda::bayesian::cudaLangevin::updateThetaBatched<double>(
                    regrid<double, 2>(theta), alpha1, regrid<const double, 2>(prior_gradient),
                    alpha2, regrid<const double, 2>(datalikelihood_gradient),
                    half_epsilon_t, regrid<const double, 2>(eta_t));
            } else if (format.size() == 1 && format[0] == 'f') {
                altar::cuda::bayesian::cudaLangevin::updateThetaBatched<float>(
                    regrid<float, 2>(theta), static_cast<float>(alpha1), regrid<const float, 2>(prior_gradient),
                    static_cast<float>(alpha2), regrid<const float, 2>(datalikelihood_gradient),
                    static_cast<float>(half_epsilon_t), regrid<const float, 2>(eta_t));
            } else {
                throw py::value_error("cudaLangevin_updateThetaBatched: unsupported grid cell type '" + format + "'");
            }
            cudaCheckError("cudaLangevin_updateThetaBatched");
            synchronize("cudaLangevin_updateThetaBatched");
        },
        "theta"_a, "alpha1"_a, "prior_gradient"_a, "alpha2"_a, "datalikelihood_gradient"_a,
        "half_epsilon_t"_a, "eta_t"_a,
        "theta += half_epsilon_t * (alpha1*prior_gradient + alpha2*datalikelihood_gradient) + eta_t");
}


// end of file
