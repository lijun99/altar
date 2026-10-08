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
from ..statistics import weighted_covariance, condition_covariance, multiplicities

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

    use_low_variance_resampler = altar.properties.bool(default=False)
    use_low_variance_resampler.doc = "whether to equal spaced random numbers for resampling"

    # public data
    cov: float = 0.0 # the actual value for COV we were able to attain


    # protocol obligations
    @altar.export
    def initialize(self, application: Application) -> typing.Self:
        """
        Initialize me and my parts given an {application} context
        """
        # the random number generator, for resampling
        self.rng = application.rng.rng
        # initialize my solver
        self.solver.initialize(application=application, scheduler=self)
        # grab the info channel
        self.info = application.info
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
        # resampling according to their likelihood
        if β > self.beta_resampling_start:
            step.weighted_theta = getattr(step, "theta_sampling", step.theta).copy()
            θ, (prior, data, posterior), θ_sampling, jacobian = self.resampling(step=step)
            # update the step after the resampling
            step.prior[...] = prior
            step.data[...] = data
            step.theta[...] = θ
            # a reparameterized step keeps theta_sampling/jacobian as separate buffers (see
            # {altar.bayesian.states.CoolingStep}), not aliases of theta -- they must be
            # reordered by the exact same sample indices, or theta/theta_sampling end up
            # describing different chains after resampling (row i's physical theta no longer
            # corresponds to row i's sampling-space theta), silently corrupting any
            # reparameterized gradient-based sampler (e.g. HMC) that reads both
            if getattr(step, 'has_reparametrization', False):
                step.theta_sampling[...] = θ_sampling
                if jacobian is not None and step.jacobian is not None:
                    step.jacobian[...] = jacobian

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
        # compute {δβ} and the normalized {w}
        β, self.cov = self.solver.solve(step.data, w)
        # publish weights on the step so all components (proposal, etc.) can consume them
        step.weights = w
        # adjust β if it is too small
        β = max(β, self.beta_min)
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
        counts = self.compute_sample_multiplicities(step=step)
        # the old samples by decreasing multiplicity, each duplicated by its count
        order = numpy.argsort(counts, kind="stable")[::-1]
        rows = numpy.repeat(order, counts[order])
        return step.theta[rows], (step.prior[rows], step.data[rows], step.posterior[rows])


    # important resampling: rank is not needed, shuffle is recommended
    def resampling(self, step: CoolingStep) -> tuple[numpy.ndarray, tuple[numpy.ndarray, ...],
                                                      numpy.ndarray | None, numpy.ndarray | None]:
        """
        Rebuild the sample and its statistics, resampled by their weights, in random order
        """
        counts = self.compute_sample_multiplicities(step=step)
        # the index of the old sample behind each new one, duplicated by its count, shuffled
        rows = numpy.repeat(numpy.arange(counts.size), counts)
        self.rng.shuffle(rows)

        self.info.log(f"resampling: unique samples {numpy.count_nonzero(counts)} out of {counts.size}")

        # a reparameterized step carries theta_sampling/jacobian as separate buffers from
        # theta (not aliases; see {altar.bayesian.states.CoolingStep}), so they must be
        # reordered by the exact same sample indices, or theta/theta_sampling end up
        # describing different chains after resampling
        θ_sampling = jacobian = None
        if getattr(step, 'has_reparametrization', False):
            θ_sampling = step.theta_sampling[rows]
            if step.jacobian is not None:
                jacobian = step.jacobian[rows]

        # return the shuffled data
        likelihoods = step.prior[rows], step.data[rows], step.posterior[rows]
        return step.theta[rows], likelihoods, θ_sampling, jacobian


    # implementation details
    def condition_covariance(self, Σ: numpy.ndarray) -> numpy.ndarray:
        """
        Make sure the covariance matrix Σ is symmetric and positive definite, replacing
        negative or small eigenvalues with min_eigenvalue_ratio*max_eigenvalue
        """
        return condition_covariance(Σ=Σ, ratio=self.min_eigenvalue_ratio)


    def compute_sample_multiplicities(self, step: CoolingStep) -> numpy.ndarray:
        """
        How many copies of each sample to keep, given the normalized weights of this cooling
        step
        """
        return multiplicities(
            w=step.weights, rng=self.rng, low_variance=self.use_low_variance_resampler)


    # private data
    rng: numpy.random.Generator
    info: journal.info


# end of file
