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
#include "metropolis.h"

// the kernel launchers
#include <altar/cuda/bayesian/cudaMetropolis.h>


auto
altar::cuda::extensions::metropolis::__init__(py::module & m) -> void
{
    auto metropolis = m.def_submodule("metropolis", "the metropolis-hastings accept/reject step");

    // compact the indices of the samples not flagged in {invalid} into the front of
    // {valid_sample_indices}, and leave the resulting count in {valid_samples[0]} (a device
    // scalar; reading it through the grid's buffer waits for the launch)
    metropolis.def(
        "cudaMetropolis_setValidSampleIndices",
        [](grid_t & valid_sample_indices, grid_t & invalid, grid_t & valid_samples) -> void {
            altar::cuda::bayesian::cudaMetropolis::setValidSampleIndices(
                regrid<int, 1>(valid_sample_indices), regrid<const int, 1>(invalid),
                regrid<int, 1>(valid_samples));
            cudaCheckError("cudaMetropolis_setValidSampleIndices");
            synchronize("cudaMetropolis_setValidSampleIndices");
        },
        "valid_sample_indices"_a, "invalid"_a, "valid_samples"_a,
        "compact the not-invalid indices to the front of valid_sample_indices; "
        "valid_samples[0] gets the resulting count");

    // gather the first {samples} valid proposals into {theta_candidate}:
    // theta_candidate[s, :] = theta_proposal[valid_sample_indices[s], :]
    metropolis.def(
        "cudaMetropolis_queueValidSamples",
        [](grid_t & theta_candidate, grid_t & theta_proposal, grid_t & valid_sample_indices,
           std::size_t samples) -> void {
            auto format = theta_proposal.view().format;
            if (format.size() == 1 && format[0] == 'd') {
                altar::cuda::bayesian::cudaMetropolis::queueValidSamples<double>(
                    regrid<double, 2>(theta_candidate), regrid<const double, 2>(theta_proposal),
                    regrid<const int, 1>(valid_sample_indices), samples);
            } else if (format.size() == 1 && format[0] == 'f') {
                altar::cuda::bayesian::cudaMetropolis::queueValidSamples<float>(
                    regrid<float, 2>(theta_candidate), regrid<const float, 2>(theta_proposal),
                    regrid<const int, 1>(valid_sample_indices), samples);
            } else {
                throw py::value_error(
                    "cudaMetropolis_queueValidSamples: unsupported grid cell type '" + format + "'");
            }
            cudaCheckError("cudaMetropolis_queueValidSamples");
            synchronize("cudaMetropolis_queueValidSamples");
        },
        "theta_candidate"_a, "theta_proposal"_a, "valid_sample_indices"_a, "samples"_a,
        "theta_candidate[s, :] = theta_proposal[valid_sample_indices[s], :]");

    // one Metropolis-Hastings accept/reject test per valid sample: on acceptance, overwrite
    // the corresponding row of {theta}/{prior}/{data}/{posterior} (indexed by
    // {valid_sample_indices[s]}) with the candidate's, and set {acceptance_flag[s] = 1}
    metropolis.def(
        "cudaMetropolis_metropolisUpdate",
        [](grid_t & theta, grid_t & prior, grid_t & data, grid_t & posterior,
           grid_t & theta_candidate, grid_t & prior_candidate, grid_t & data_candidate,
           grid_t & posterior_candidate, grid_t & dices, grid_t & acceptance_flag,
           grid_t & valid_sample_indices, std::size_t batch) -> void {
            auto format = theta.view().format;
            if (format.size() == 1 && format[0] == 'd') {
                altar::cuda::bayesian::cudaMetropolis::metropolisUpdate<double>(
                    regrid<double, 2>(theta), regrid<double, 1>(prior),
                    regrid<double, 1>(data), regrid<double, 1>(posterior),
                    regrid<const double, 2>(theta_candidate), regrid<const double, 1>(prior_candidate),
                    regrid<const double, 1>(data_candidate), regrid<const double, 1>(posterior_candidate),
                    regrid<const double, 1>(dices), regrid<int, 1>(acceptance_flag),
                    regrid<const int, 1>(valid_sample_indices), batch);
            } else if (format.size() == 1 && format[0] == 'f') {
                altar::cuda::bayesian::cudaMetropolis::metropolisUpdate<float>(
                    regrid<float, 2>(theta), regrid<float, 1>(prior),
                    regrid<float, 1>(data), regrid<float, 1>(posterior),
                    regrid<const float, 2>(theta_candidate), regrid<const float, 1>(prior_candidate),
                    regrid<const float, 1>(data_candidate), regrid<const float, 1>(posterior_candidate),
                    regrid<const float, 1>(dices), regrid<int, 1>(acceptance_flag),
                    regrid<const int, 1>(valid_sample_indices), batch);
            } else {
                throw py::value_error(
                    "cudaMetropolis_metropolisUpdate: unsupported grid cell type '" + format + "'");
            }
            cudaCheckError("cudaMetropolis_metropolisUpdate");
            synchronize("cudaMetropolis_metropolisUpdate");
        },
        "theta"_a, "prior"_a, "data"_a, "posterior"_a,
        "theta_candidate"_a, "prior_candidate"_a, "data_candidate"_a, "posterior_candidate"_a,
        "dices"_a, "acceptance_flag"_a, "valid_sample_indices"_a, "batch"_a,
        "accept/reject each of the first {batch} candidates; on acceptance, overwrite the "
        "corresponding row (indexed by valid_sample_indices) in place");

    // after {cudaMetropolis_metropolisUpdate}, for a walk in sampling space: copy the
    // sampling-space row and log-jacobian of each accepted candidate
    metropolis.def(
        "cudaMetropolis_updateSampling",
        [](grid_t & theta_sampling, grid_t & jacobian, grid_t & theta_sampling_candidate,
           grid_t & jacobian_candidate, grid_t & acceptance_flag, grid_t & valid_sample_indices,
           std::size_t batch) -> void {
            auto format = theta_sampling.view().format;
            if (format.size() == 1 && format[0] == 'd') {
                altar::cuda::bayesian::cudaMetropolis::updateSampling<double>(
                    regrid<double, 2>(theta_sampling), regrid<double, 1>(jacobian),
                    regrid<const double, 2>(theta_sampling_candidate),
                    regrid<const double, 1>(jacobian_candidate),
                    regrid<const int, 1>(acceptance_flag),
                    regrid<const int, 1>(valid_sample_indices), batch);
            } else if (format.size() == 1 && format[0] == 'f') {
                altar::cuda::bayesian::cudaMetropolis::updateSampling<float>(
                    regrid<float, 2>(theta_sampling), regrid<float, 1>(jacobian),
                    regrid<const float, 2>(theta_sampling_candidate),
                    regrid<const float, 1>(jacobian_candidate),
                    regrid<const int, 1>(acceptance_flag),
                    regrid<const int, 1>(valid_sample_indices), batch);
            } else {
                throw py::value_error(
                    "cudaMetropolis_updateSampling: unsupported grid cell type '" + format + "'");
            }
            cudaCheckError("cudaMetropolis_updateSampling");
            synchronize("cudaMetropolis_updateSampling");
        },
        "theta_sampling"_a, "jacobian"_a, "theta_sampling_candidate"_a, "jacobian_candidate"_a,
        "acceptance_flag"_a, "valid_sample_indices"_a, "batch"_a,
        "copy the sampling-space row and log-jacobian of each of the first {batch} candidates "
        "that was accepted into the row valid_sample_indices names");
}


// end of file
