# -*- python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#


# externals
from __future__ import annotations
import typing
import numpy
# the package
import altar
# my protocol
from .Scheduler import Scheduler as scheduler
from ..statistics import weighted_covariance, condition_covariance

if typing.TYPE_CHECKING:
    import journal
    from altar.bayesian.states.CoolingStep import CoolingStep
    from altar.shells.Application import Application

# a step's sample matrix and its prior, data and posterior log likelihoods
Resampled = tuple[numpy.ndarray, tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]]


# declaration
class COV(altar.component, family="altar.schedulers.cov", implements=scheduler):
    r"""
    Annealing schedule based on attaining a particular value for the coefficient of variation
    (COV) of the data likelihood; after Ching[2007].

    The goal is to compute a proposed update δβ_m to the temperature β_m such that the vector
    of weights w_m given by

        w_m := π(D|θ_m)^{δβ_m}

    has a particular target value for

        COV(w_m) := \sqrt{<(w_m-<w_m>)^2>} / <w_m>
    """

    # user configurable state
    target = altar.properties.float(default=1.0)
    target.doc = 'the target value for COV'

    solver = altar.bayesian.solver()
    solver.doc = 'the δβ solver'

    check_positive_definiteness = altar.properties.bool(default=True)
    check_positive_definiteness.doc = 'whether to check the positive definiteness of Σ matrix and condition it accordingly'

    min_eigenvalue_ratio = altar.properties.float(default=0.001)
    min_eigenvalue_ratio.doc = 'the desired minimal eigenvalue of Σ matrix, as a ratio to the max eigenvalue'

    beta_resampling_start = altar.properties.float(default=0)
    beta_resampling_start.doc = 'the beta threshold to start the resampling procedure'

    beta_min = altar.properties.float(default=0)
    beta_min.doc = 'the minimum beta value to be used'

    resampler = altar.bayesian.resampler()
    resampler.doc = "replaces the weighted samples by equally weighted ones, by default " \
                    "{altar.bayesian.importanceresampler} with systematic resampling"

    # public data
    cov: float = 0.0 # the actual value for COV we were able to attain
    log_evidence: float = 0.0 # the running estimate of log p(d), the sum of log <w_m> over the steps


    # protocol obligations
    @altar.export
    def initialize(self, application: Application) -> typing.Self:
        """
        Initialize me and my parts given an {application} context
        """
        # initialize my resampler
        self.resampler.initialize(application=application)
        # and my solver
        self.solver.initialize(application=application, scheduler=self)
        # grab the info channel
        self.info = application.info
        # the evidence accumulates from here; a controller may seed it, e.g. cross-fade sampling
        self.log_evidence = 0.0
        # all done
        return self


    @altar.export
    def update(self, step: CoolingStep) -> CoolingStep:
        """
        Push {step} forward along the annealing schedule
        """

        # get the new temperature and store it
        β = self.update_temperature(step=step)
        # the samples the weights belong to, for the proposal covariance: those before resampling
        step.weighted_theta = None
        # the rows the resampling keeps, if it happens
        step.ancestors = None
        # resampling according to their likelihood
        if β > self.beta_resampling_start:
            step.weighted_theta = getattr(step, "theta_sampling", step.theta).copy()
            # the sample behind each new one, published for the diagnostics of the run; every
            # part of the state follows the same rows, the samples in sampling and physical
            # space, the jacobian and the densities
            step.ancestors = self.resampler.rows(w=step.weights)
            step.reorder(rows=step.ancestors)

        # update the step (common procedures with or w/o resampling)
        step.beta = β

        # recompute posterior with updated beta
        step.compute_posterior()

        # and return it
        return step


    @altar.export
    def update_temperature(self, step: CoolingStep) -> float:
        """
        Generate the next temperature increment
        """
        # the normalized weights, filled in by the solver
        w = numpy.zeros(step.samples)
        # start from the temperature of {step}, which a restarted run did not reach through me
        self.solver.beta = step.beta
        # compute {δβ} and the normalized {w}
        β, self.cov = self.solver.solve(step.data, w)
        # publish weights on the step so all components (proposal, etc.) can consume them
        step.weights = w
        # adjust β if it is too small
        β = max(β, self.beta_min)
        # the evidence, log p(d) = Σ_m log <w_m> (Ching and Chen, 2007), with the unnormalized
        # weights w_m = exp(δβ_m llk)
        δβ = β - step.beta
        llk = numpy.asarray(step.data, dtype=numpy.float64)
        top = llk.max()
        self.log_evidence += δβ * top + numpy.log(numpy.mean(numpy.exp(δβ * (llk - top))))
        # and return the new temperature
        return β


    @altar.export
    def compute_covariance(self, step: CoolingStep) -> numpy.ndarray:
        r"""
        Compute the parameter covariance Σ of the sample in {step}

          Σ = c_m^2 \sum_{i \in samples} \tilde{w}_{i} θ_i θ_i^T} - \bar{θ} \bar{θ}^Τ

        where

          \bar{θ} = \sum_{i \in samples} \tilde{w}_{i} θ_{i}

        The covariance Σ gets used to build a proposal pdf for the posterior
        """
        # unpack what i need
        w = step.weights # published by update_temperature(); assumed normalized
        θ = step.theta # the current sample set

        # check the geometries
        assert w.shape == (step.samples,)
        assert θ.shape == (step.samples, step.parameters)

        # the weighted outer products about the weighted mean, in one matrix product
        Σ = weighted_covariance(θ, w)

        # condition the covariance matrix
        if self.check_positive_definiteness:
            Σ = self.condition_covariance(Σ=Σ)

        # all done
        return Σ


    @altar.export
    def rank(self, step: CoolingStep) -> Resampled:
        """
        Rebuild the sample and its statistics sorted by the likelihood of the parameter values
        """
        counts = self.resampler.multiplicities(w=step.weights)
        # the old samples by decreasing multiplicity, each duplicated by its count
        order = numpy.argsort(counts, kind="stable")[::-1]
        rows = numpy.repeat(order, counts[order])
        return step.theta[rows], (step.prior[rows], step.data[rows], step.posterior[rows])


    # implementation details
    def condition_covariance(self, Σ: numpy.ndarray) -> numpy.ndarray:
        """
        Make sure the covariance matrix Σ is symmetric and positive definite, replacing
        negative or small eigenvalues with min_eigenvalue_ratio*max_eigenvalue
        """
        return condition_covariance(Σ=Σ, ratio=self.min_eigenvalue_ratio)


    # private data
    info: journal.info


# end of file
