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
import types
import typing
import numpy
# the package
import altar
# my protocol
from .Proposal import Proposal as proposal
from ..statistics import weighted_covariance, condition_covariance

if typing.TYPE_CHECKING:
    import journal
    from altar.bayesian.controllers.Annealer import Annealer
    from altar.bayesian.states.BayesianState import BayesianState
    from altar.shells.Application import Application
    from altar.simulations.Archiver import Archiver

# declaration
class GaussianProposal(altar.component, family="altar.proposals.gaussian", implements=proposal):
    """
    Gaussian proposal kernel N(θ, Σ) for MCMC sampling.

    The proposal covariance Σ is either fixed (set via {set_sigma}) or periodically
    recomputed from the weighted auto-correlation of the current sample set {θ}:

        Σ = Σ_i w_i (θ_i - θ̄)(θ_i - θ̄)^T

    where w_i are importance weights.  In a tempering/annealing scheme the weights are

        w_i ∝ exp(Δβ · log p(data|θ_i))

    as provided by the annealer scheduler after each temperature step.
    """

    # user configurable state
    check_positive_definiteness = altar.properties.bool(default=True)
    check_positive_definiteness.doc = 'whether to check the positive definiteness of Σ and condition it'

    min_eigenvalue_ratio = altar.properties.float(default=0.001)
    min_eigenvalue_ratio.doc = 'minimum eigenvalue of Σ as a ratio to the max eigenvalue'

    update_interval = altar.properties.int(default=1)
    update_interval.doc = (
        'recompute Σ from weighted theta auto-correlation every this many annealing steps; '
        '0 = compute once at initialization, never auto-update afterwards'
    )

    archive_sigma = altar.properties.bool(default=True)
    archive_sigma.doc = 'whether to record the proposal covariance Σ at each archiver save point'

    # protocol obligations
    @altar.export
    def initialize(self, application: Application) -> typing.Self:
        """
        Initialize me and my parts given an {application} context
        """
        # grab the info channel
        self.info = application.info
        # and the random number generator
        self.rng = application.rng.rng

        # register with the archiver so our record() is called at each save point; the
        # archiver lives on the controller, not the application itself, and by this point
        # the controller has already initialized it (see Annealer.initialize())
        if self.archive_sigma:
            archiver = getattr(getattr(application, 'controller', None), 'archiver', None)
            if archiver is not None:
                archiver.register(self)

        # all done
        return self

    def set_sigma(self, sigma: numpy.ndarray) -> typing.Self:
        """
        Fix the proposal covariance to the given matrix, disabling auto-update.
        Call before the first {propose} to use a user-supplied Σ rather than computing
        it from sample auto-correlation.
        """
        self._sigma = numpy.array(sigma, dtype=float)
        self._sigma_is_fixed = True
        # invalidate the cached decomposition so it is rebuilt on the next propose
        self._sigma_chol = None
        # all done
        return self

    @altar.export
    def propose(self, sampler: typing.Any, step: BayesianState,
                annealer: Annealer | None = None) -> numpy.ndarray:
        """
        Propose a new sample set using a Gaussian random walk with covariance Σ
        """
        # update Σ and its Cholesky decomposition if needed
        if self._needs_prepare(step=step, scaling=sampler.scaling):
            self._prepare(sampler=sampler, step=step, annealer=annealer)

        # generate the displaced samples
        return self._displace(sample=step.theta)


    @altar.export
    def new_walk(self) -> typing.Self:
        """
        A walk of the chains is about to start: recompute Σ from the current samples, when it
        is not fixed, before the next proposal
        """
        self._walk_pending = True
        return self


    # implementation details
    def _prepare(self, sampler: typing.Any, step: BayesianState,
                 annealer: Annealer | None) -> typing.Self:
        """
        Update Σ if warranted, then scale it by {sampler.scaling}^2 and Cholesky-decompose
        """
        dispatcher = annealer.dispatcher if annealer is not None else None
        if dispatcher is not None:
            dispatcher.notify(event=dispatcher.prepare_sampling_pdf_start, controller=annealer)

        # detect the start of a new walk of the chains
        new_walk = self._walk_pending

        # auto-update Σ when not fixed
        if not self._sigma_is_fixed:
            if self._sigma is None:
                # first-time initialization: always compute
                self._sigma = self._compute_sigma(step=step, annealer=annealer)
            elif new_walk:
                self._anneal_count += 1
                # recompute every update_interval walks (0 means never after first)
                if self.update_interval > 0 and (self._anneal_count % self.update_interval == 0):
                    self._sigma = self._compute_sigma(step=step, annealer=annealer)

        # scale Σ by the sampler scaling factor and Cholesky-decompose for sampling
        self._sigma_chol = numpy.linalg.cholesky(self._sigma * sampler.scaling ** 2)

        # cache preparation state
        self._walk_pending = False
        self._prepared_scaling = sampler.scaling

        if dispatcher is not None:
            dispatcher.notify(event=dispatcher.prepare_sampling_pdf_finish, controller=annealer)

        # all done
        return self


    def _needs_prepare(self, step: BayesianState, scaling: float) -> bool:
        """
        Decide whether we need to refresh the cached Cholesky decomposition (and possibly Σ)
        """
        # not yet initialized
        if self._sigma_chol is None:
            return True
        # scaling changed: must re-decompose even with fixed Σ
        if self._prepared_scaling != scaling:
            return True
        # for auto-update: trigger at the start of every walk
        if not self._sigma_is_fixed and self._walk_pending:
            return True
        return False


    def _compute_sigma(self, step: BayesianState, annealer: Annealer | None) -> numpy.ndarray:
        """
        Compute Σ from the importance-weighted auto-correlation of {step.theta}
        """
        weights = self._get_weights(step=step, annealer=annealer)
        # the weights belong to the samples before resampling, when the scheduler kept them
        weighted = getattr(step, "weighted_theta", None)
        if weighted is not None:
            samples, parameters = weighted.shape
            step = types.SimpleNamespace(theta=weighted, samples=samples, parameters=parameters)
        return self.compute_covariance(step=step, w=weights)


    def _get_weights(self, step: BayesianState, annealer: Annealer | None) -> numpy.ndarray:
        """
        Return importance weights for the covariance computation.
        In a tempering scheme these are w_i ∝ exp(Δβ · data_i), written to step.weights
        by the scheduler after each temperature update.  Falls back to uniform weights
        when not yet set.
        """
        samples = step.samples
        w = getattr(step, 'weights', None)
        if w is not None:
            return w
        # uniform fallback (beta=0 or no scheduler)
        return numpy.full(samples, 1.0 / samples)


    def _displace(self, sample: numpy.ndarray) -> numpy.ndarray:
        """
        The samples displaced by the Gaussian random walk, sample + L z with z ~ N(0, 1) and
        L the Cholesky factor of the scaled covariance, in the precision of the samples
        """
        z = self.rng.standard_normal(size=sample.shape)
        return (sample + z @ self._sigma_chol.T).astype(sample.dtype, copy=False)

    @altar.export
    def compute_covariance(self, step: typing.Any, w: numpy.ndarray) -> numpy.ndarray:
        r"""
        Compute the parameter covariance Σ of the sample in {step} with weights {w}:

            Σ = Σ_i w_i θ_i θ_i^T - θ̄ θ̄^T,   θ̄ = Σ_i w_i θ_i

        The weights {w} are importance weights; in a tempering scheme,
            w_i ∝ exp(Δβ · log p(data|θ_i))
        """
        θ = step.theta
        samples = step.samples
        parameters = step.parameters

        assert w.shape == (samples,)
        assert θ.shape == (samples, parameters)

        # the weighted outer products about the weighted mean, in one matrix product
        Σ = weighted_covariance(θ, w)

        # condition the covariance matrix if requested
        if self.check_positive_definiteness:
            Σ = self.condition_covariance(Σ=Σ)

        return Σ

    def condition_covariance(self, Σ: numpy.ndarray) -> numpy.ndarray:
        """
        Ensure Σ is symmetric positive definite by lifting small/negative eigenvalues
        """
        return condition_covariance(Σ=Σ, ratio=self.min_eigenvalue_ratio)

    @altar.export
    def record(self, archiver: Archiver) -> typing.Self:
        """
        Record the proposal covariance Σ via {archiver}.
        Called automatically at each save point when registered during initialize().
        """
        if self._sigma is not None:
            archiver.write("Proposal/sigma", self._sigma,
                           {"fixed": self._sigma_is_fixed,
                            "update_interval": self.update_interval})
        return self

    # private data
    info: journal.info | None = None
    rng: numpy.random.Generator

    # owned proposal covariance and its Cholesky factor
    _sigma: numpy.ndarray | None = None       # the unscaled proposal covariance Σ
    _sigma_chol: numpy.ndarray | None = None  # lower Cholesky factor of (scaling^2 * Σ)
    _sigma_is_fixed: bool = False  # True if sigma was set externally via set_sigma()

    # update tracking
    _anneal_count: int = 0      # number of walks since the first Σ computation
    _walk_pending: bool = True  # whether a walk started since _prepare was last called
    _prepared_scaling: float | None = None  # scaling at which _prepare was last called

# end of file
