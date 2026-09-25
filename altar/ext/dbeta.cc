// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//

// external dependencies
#include "external.h"
// namespace setup
#include "forward.h"


void
altar::py::dbeta(py::module & m)
{
    // the COV class: manages the annealing schedule state (beta, cov) across a run
    py::class_<altar::bayesian::COV>(m, "COV", "the annealing schedule state manager")

        // constructor: the python-facing argument order is (rng, maxiter, tolerance,
        // target), but the C++ constructor takes (rng, tolerance, maxIterations, target) --
        // preserve the existing python call shape by reordering here
        .def(
            py::init([](gsl_rng & rng, size_t maxiter, double tolerance, double target) {
                return new altar::bayesian::COV(&rng, tolerance, maxiter, target);
            }),
            "rng"_a, "maxiter"_a, "tolerance"_a, "target"_a,
            "allocate a COV instance to manage the annealing schedule")

        // accessors
        .def_property_readonly(
            "beta", &altar::bayesian::COV::beta, "the current value of the annealing temperature")
        .def_property_readonly(
            "cov", &altar::bayesian::COV::cov, "the coefficient of variation attained by the last update")

        // the two delta-beta solvers
        .def(
            "dbeta_brent",
            [](altar::bayesian::COV & self, gsl_vector & llk, double llkMedian, gsl_vector & w)
                -> std::tuple<double, double> {
                self.dbeta_brent(&llk, llkMedian, &w);
                return { self.beta(), self.cov() };
            },
            "llk"_a, "llkMedian"_a, "w"_a,
            "compute the next increment to the annealing temperature using the Brent "
            "algorithm from GSL; returns (beta, cov)")

        .def(
            "dbeta_grid",
            [](altar::bayesian::COV & self, gsl_vector & llk, double llkMedian, gsl_vector & w)
                -> std::tuple<double, double> {
                self.dbeta_grid(&llk, llkMedian, &w);
                return { self.beta(), self.cov() };
            },
            "llk"_a, "llkMedian"_a, "w"_a,
            "compute the next increment to the annealing temperature using an iterative "
            "grid search; returns (beta, cov)");

    // low variance random generator for importance resampling
    m.def(
        "low_variance_random",
        [](gsl_rng & rng, gsl_vector & r) -> void {
            // get the length of the vector
            size_t n = r.size;
            double step = 1.0 / double(n);

            // generate a random number (starting position)
            double s = gsl_rng_uniform(&rng) * step;

            // generate the equal spacing sequence
            for (size_t i = 0; i < n; ++i) {
                gsl_vector_set(&r, i, s);
                s += step;
            }

            // all done
            return;
        },
        "rng"_a, "r"_a,
        "low variance random generator for importance resampling");

    return;
}

// end of file
