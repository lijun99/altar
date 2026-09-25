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
#include "leapfrog.h"

// the kernel launchers
#include <altar/cuda/bayesian/cudaLeapfrog.h>


auto
altar::cuda::extensions::leapfrog::__init__(py::module & m) -> void
{
    auto leapfrog = m.def_submodule("leapfrog", "the leapfrog integrator for hamiltonian monte carlo");

    // fill {momentum} with iid draws from N(0, 1)
    leapfrog.def(
        "cudaLeapfrog_sampleMomentum",
        [](grid_t & momentum) -> void {
            auto format = momentum.view().format;
            if (format.size() == 1 && format[0] == 'd') {
                altar::cuda::bayesian::cudaLeapfrog::sampleMomentum<double>(regrid<double, 2>(momentum));
            } else if (format.size() == 1 && format[0] == 'f') {
                altar::cuda::bayesian::cudaLeapfrog::sampleMomentum<float>(regrid<float, 2>(momentum));
            } else {
                throw py::value_error("cudaLeapfrog_sampleMomentum: unsupported grid cell type '" + format + "'");
            }
            cudaCheckError("cudaLeapfrog_sampleMomentum");
            synchronize("cudaLeapfrog_sampleMomentum");
        },
        "momentum"_a, "fill momentum with iid draws from N(0, 1)");

    // potential[s] = -(prior[s] + beta*data[s]); grad_potential[s,:] =
    // -(grad_prior[s,:] + beta*grad_data[s,:]), or, with {jacobian} given (not None),
    // -(grad_prior[s,:] + beta*jacobian[s,:]*grad_data[s,:])
    leapfrog.def(
        "cudaLeapfrog_computePotentialAndGradient",
        [](grid_t & prior, grid_t & data, grid_t & grad_prior, grid_t & grad_data,
           grid_t & potential, grid_t & grad_potential, double beta,
           py::object jacobian) -> void {
            auto format = prior.view().format;
            bool reparam = !jacobian.is_none();
            if (format.size() == 1 && format[0] == 'd') {
                if (reparam) {
                    altar::cuda::bayesian::cudaLeapfrog::computePotentialAndGradientReparam<double>(
                        regrid<const double, 1>(prior), regrid<const double, 1>(data),
                        regrid<const double, 2>(grad_prior), regrid<const double, 2>(grad_data),
                        regrid<const double, 2>(jacobian.cast<grid_t &>()),
                        regrid<double, 1>(potential), regrid<double, 2>(grad_potential), beta);
                } else {
                    altar::cuda::bayesian::cudaLeapfrog::computePotentialAndGradient<double>(
                        regrid<const double, 1>(prior), regrid<const double, 1>(data),
                        regrid<const double, 2>(grad_prior), regrid<const double, 2>(grad_data),
                        regrid<double, 1>(potential), regrid<double, 2>(grad_potential), beta);
                }
            } else if (format.size() == 1 && format[0] == 'f') {
                auto betaf = static_cast<float>(beta);
                if (reparam) {
                    altar::cuda::bayesian::cudaLeapfrog::computePotentialAndGradientReparam<float>(
                        regrid<const float, 1>(prior), regrid<const float, 1>(data),
                        regrid<const float, 2>(grad_prior), regrid<const float, 2>(grad_data),
                        regrid<const float, 2>(jacobian.cast<grid_t &>()),
                        regrid<float, 1>(potential), regrid<float, 2>(grad_potential), betaf);
                } else {
                    altar::cuda::bayesian::cudaLeapfrog::computePotentialAndGradient<float>(
                        regrid<const float, 1>(prior), regrid<const float, 1>(data),
                        regrid<const float, 2>(grad_prior), regrid<const float, 2>(grad_data),
                        regrid<float, 1>(potential), regrid<float, 2>(grad_potential), betaf);
                }
            } else {
                throw py::value_error(
                    "cudaLeapfrog_computePotentialAndGradient: unsupported grid cell type '" + format + "'");
            }
            cudaCheckError("cudaLeapfrog_computePotentialAndGradient");
            synchronize("cudaLeapfrog_computePotentialAndGradient");
        },
        "prior"_a, "data"_a, "grad_prior"_a, "grad_data"_a, "potential"_a, "grad_potential"_a,
        "beta"_a, "jacobian"_a = py::none(),
        "potential = -(prior + beta*data); grad_potential = -(grad_prior + beta*grad_data), "
        "scaled elementwise by {jacobian} when given");

    // kinetic[s] = 0.5 * ||momentum[s, :]||^2
    leapfrog.def(
        "cudaLeapfrog_kineticEnergy",
        [](grid_t & momentum, grid_t & kinetic) -> void {
            auto format = momentum.view().format;
            if (format.size() == 1 && format[0] == 'd') {
                altar::cuda::bayesian::cudaLeapfrog::kineticEnergy<double>(
                    regrid<const double, 2>(momentum), regrid<double, 1>(kinetic));
            } else if (format.size() == 1 && format[0] == 'f') {
                altar::cuda::bayesian::cudaLeapfrog::kineticEnergy<float>(
                    regrid<const float, 2>(momentum), regrid<float, 1>(kinetic));
            } else {
                throw py::value_error("cudaLeapfrog_kineticEnergy: unsupported grid cell type '" + format + "'");
            }
            cudaCheckError("cudaLeapfrog_kineticEnergy");
            synchronize("cudaLeapfrog_kineticEnergy");
        },
        "momentum"_a, "kinetic"_a, "kinetic[s] = 0.5 * ||momentum[s, :]||^2");

    // theta += step * momentum, elementwise, in place
    leapfrog.def(
        "cudaLeapfrog_updatePosition",
        [](grid_t & theta, grid_t & momentum, double step) -> void {
            auto format = theta.view().format;
            if (format.size() == 1 && format[0] == 'd') {
                altar::cuda::bayesian::cudaLeapfrog::updatePosition<double>(
                    regrid<double, 2>(theta), regrid<const double, 2>(momentum), step);
            } else if (format.size() == 1 && format[0] == 'f') {
                altar::cuda::bayesian::cudaLeapfrog::updatePosition<float>(
                    regrid<float, 2>(theta), regrid<const float, 2>(momentum), static_cast<float>(step));
            } else {
                throw py::value_error("cudaLeapfrog_updatePosition: unsupported grid cell type '" + format + "'");
            }
            cudaCheckError("cudaLeapfrog_updatePosition");
            synchronize("cudaLeapfrog_updatePosition");
        },
        "theta"_a, "momentum"_a, "step"_a, "theta += step * momentum, in place");

    // momentum += scale * gradU, elementwise, in place
    leapfrog.def(
        "cudaLeapfrog_updateMomentum",
        [](grid_t & momentum, grid_t & gradU, double scale) -> void {
            auto format = momentum.view().format;
            if (format.size() == 1 && format[0] == 'd') {
                altar::cuda::bayesian::cudaLeapfrog::updateMomentum<double>(
                    regrid<double, 2>(momentum), regrid<const double, 2>(gradU), scale);
            } else if (format.size() == 1 && format[0] == 'f') {
                altar::cuda::bayesian::cudaLeapfrog::updateMomentum<float>(
                    regrid<float, 2>(momentum), regrid<const float, 2>(gradU), static_cast<float>(scale));
            } else {
                throw py::value_error("cudaLeapfrog_updateMomentum: unsupported grid cell type '" + format + "'");
            }
            cudaCheckError("cudaLeapfrog_updateMomentum");
            synchronize("cudaLeapfrog_updateMomentum");
        },
        "momentum"_a, "gradU"_a, "scale"_a, "momentum += scale * gradU, in place");

    // one Metropolis-Hastings accept/reject test per sample: mask[s] = 1 if
    // log(u) < -deltaH[s] for a fresh uniform draw u, else 0
    leapfrog.def(
        "cudaLeapfrog_metropolis",
        [](grid_t & deltaH, grid_t & mask) -> void {
            auto format = deltaH.view().format;
            if (format.size() == 1 && format[0] == 'd') {
                altar::cuda::bayesian::cudaLeapfrog::metropolis<double>(
                    regrid<const double, 1>(deltaH), regrid<int, 1>(mask));
            } else if (format.size() == 1 && format[0] == 'f') {
                altar::cuda::bayesian::cudaLeapfrog::metropolis<float>(
                    regrid<const float, 1>(deltaH), regrid<int, 1>(mask));
            } else {
                throw py::value_error("cudaLeapfrog_metropolis: unsupported grid cell type '" + format + "'");
            }
            cudaCheckError("cudaLeapfrog_metropolis");
            synchronize("cudaLeapfrog_metropolis");
        },
        "deltaH"_a, "mask"_a, "mask[s] = 1 if log(u) < -deltaH[s] for a fresh uniform draw u");

    // restore both {theta}/{momentum} to their {*_old} values, row by row, for every sample
    // where {mask[sample] == 0} (rejected)
    leapfrog.def(
        "cudaLeapfrog_restoreRejected",
        [](grid_t & theta, grid_t & theta_old, grid_t & momentum, grid_t & momentum_old,
           grid_t & mask) -> void {
            auto format = theta.view().format;
            if (format.size() == 1 && format[0] == 'd') {
                altar::cuda::bayesian::cudaLeapfrog::restoreRejected<double>(
                    regrid<double, 2>(theta), regrid<const double, 2>(theta_old),
                    regrid<double, 2>(momentum), regrid<const double, 2>(momentum_old),
                    regrid<const int, 1>(mask));
            } else if (format.size() == 1 && format[0] == 'f') {
                altar::cuda::bayesian::cudaLeapfrog::restoreRejected<float>(
                    regrid<float, 2>(theta), regrid<const float, 2>(theta_old),
                    regrid<float, 2>(momentum), regrid<const float, 2>(momentum_old),
                    regrid<const int, 1>(mask));
            } else {
                throw py::value_error("cudaLeapfrog_restoreRejected: unsupported grid cell type '" + format + "'");
            }
            cudaCheckError("cudaLeapfrog_restoreRejected");
            synchronize("cudaLeapfrog_restoreRejected");
        },
        "theta"_a, "theta_old"_a, "momentum"_a, "momentum_old"_a, "mask"_a,
        "restore theta/momentum rows to their _old values wherever mask == 0");

    // the single-matrix counterpart of {cudaLeapfrog_restoreRejected}
    leapfrog.def(
        "cudaLeapfrog_restoreMatrix",
        [](grid_t & current, grid_t & backup, grid_t & mask) -> void {
            auto format = current.view().format;
            if (format.size() == 1 && format[0] == 'd') {
                altar::cuda::bayesian::cudaLeapfrog::restoreMatrix<double>(
                    regrid<double, 2>(current), regrid<const double, 2>(backup), regrid<const int, 1>(mask));
            } else if (format.size() == 1 && format[0] == 'f') {
                altar::cuda::bayesian::cudaLeapfrog::restoreMatrix<float>(
                    regrid<float, 2>(current), regrid<const float, 2>(backup), regrid<const int, 1>(mask));
            } else {
                throw py::value_error("cudaLeapfrog_restoreMatrix: unsupported grid cell type '" + format + "'");
            }
            cudaCheckError("cudaLeapfrog_restoreMatrix");
            synchronize("cudaLeapfrog_restoreMatrix");
        },
        "current"_a, "backup"_a, "mask"_a, "restore current's rows to backup's wherever mask == 0");
}


// end of file
