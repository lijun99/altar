# -*- python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# lijun zhu <ljzhu@gps.caltech.edu>
#
# (c) 2013-2021 parasim inc
# (c) 2010-2021 california institute of technology
# all rights reserved
#

# the package
import altar
# my protocol
from .Proposal import Proposal as proposal

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

    # protocol obligations
    @altar.export
    def initialize(self, application):
        """
        Initialize me and my parts given an {application} context
        """
        # grab the info channel
        self.info = application.info
        # get the rng capsule
        self.rng = application.rng.rng
        # distribution for random walk displacement vectors
        self.uninormal = altar.pdf.ugaussian(rng=self.rng)

        # register with the archiver so our record() is called at each save point
        archiver = getattr(application, 'archiver', None)
        if archiver is not None:
            archiver.register(self)

        # all done
        return self

    def set_sigma(self, sigma):
        """
        Fix the proposal covariance to the given matrix, disabling auto-update.
        Call before the first {propose} to use a user-supplied Σ rather than computing
        it from sample auto-correlation.
        """
        self._sigma = sigma.clone()
        self._sigma_is_fixed = True
        # invalidate the cached decomposition so it is rebuilt on the next propose
        self._sigma_chol = None
        # all done
        return self

    @altar.export
    def propose(self, sampler, step, annealer=None):
        """
        Propose a new sample set using a Gaussian random walk with covariance Σ
        """
        # update Σ and its Cholesky decomposition if needed
        if self._needs_prepare(step=step, scaling=sampler.scaling):
            self._prepare(sampler=sampler, step=step, annealer=annealer)

        # generate the displaced samples
        return self._displace(sample=step.theta)


    # implementation details
    def _prepare(self, sampler, step, annealer):
        """
        Update Σ if warranted, then scale it by {sampler.scaling}^2 and Cholesky-decompose
        """
        dispatcher = annealer.dispatcher if annealer is not None else None
        if dispatcher is not None:
            dispatcher.notify(event=dispatcher.prepare_sampling_pdf_start, controller=annealer)

        # detect a new annealing step (beta changed)
        new_beta = (self._prepared_beta != step.beta)

        # auto-update Σ when not fixed
        if not self._sigma_is_fixed:
            if self._sigma is None:
                # first-time initialization: always compute
                self._sigma = self._compute_sigma(step=step, annealer=annealer)
            elif new_beta:
                self._anneal_count += 1
                # recompute every update_interval annealing steps (0 means never after first)
                if self.update_interval > 0 and (self._anneal_count % self.update_interval == 0):
                    self._sigma = self._compute_sigma(step=step, annealer=annealer)

        # copy to step.sigma for archiving / downstream use
        step.sigma.copy(self._sigma)

        # scale Σ by the sampler scaling factor and Cholesky-decompose for sampling
        Σ = self._sigma.clone()
        Σ *= sampler.scaling ** 2
        self._sigma_chol = altar.lapack.cholesky_decomposition(Σ)

        # cache preparation state
        self._prepared_beta = step.beta
        self._prepared_scaling = sampler.scaling

        if dispatcher is not None:
            dispatcher.notify(event=dispatcher.prepare_sampling_pdf_finish, controller=annealer)

        # all done
        return self


    def _needs_prepare(self, step, scaling):
        """
        Decide whether we need to refresh the cached Cholesky decomposition (and possibly Σ)
        """
        # not yet initialized
        if self._sigma_chol is None:
            return True
        # scaling changed: must re-decompose even with fixed Σ
        if self._prepared_scaling != scaling:
            return True
        # for auto-update: trigger on every new annealing step (beta change)
        if not self._sigma_is_fixed and self._prepared_beta != step.beta:
            return True
        return False


    def _compute_sigma(self, step, annealer):
        """
        Compute Σ from the importance-weighted auto-correlation of {step.theta}
        """
        weights = self._get_weights(step=step, annealer=annealer)
        Σ = self.compute_covariance(step=step, w=weights)
        return Σ


    def _get_weights(self, step, annealer):
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
        w = altar.vector(shape=samples).zero()
        value = 1.0 / samples
        for i in range(samples):
            w[i] = value
        return w


    def _displace(self, sample):
        """
        Construct a set of displacement vectors for the Gaussian random walk
        """
        # get the Cholesky factor of the scaled covariance
        Σ_chol = self._sigma_chol

        # build random displacement vectors; shape (parameters x samples) for convenience
        δT = altar.matrix(shape=tuple(reversed(sample.shape))).random(pdf=self.uninormal)
        # multiply by the Cholesky factor: δT ← Σ_chol · δT
        δT = altar.blas.dtrmm(
            Σ_chol.sideLeft, Σ_chol.lowerTriangular, Σ_chol.opNoTrans, Σ_chol.nonUnitDiagonal,
            1, Σ_chol, δT)

        # transpose to (samples x parameters)
        δ = altar.matrix(shape=sample.shape)
        δT.transpose(δ)
        # offset by the current sample
        δ += sample
        # and return it
        return δ

    @altar.export
    def compute_covariance(self, step, w):
        r"""
        Compute the parameter covariance Σ of the sample in {step} with weights {w}:

            Σ = Σ_i w_i θ_i θ_i^T - θ̄ θ̄^T,   θ̄ = Σ_i w_i θ_i

        The weights {w} are importance weights; in a tempering scheme,
            w_i ∝ exp(Δβ · log p(data|θ_i))
        """
        θ = step.theta
        samples = step.samples
        parameters = step.parameters

        assert w.shape == samples
        assert θ.shape == (samples, parameters)

        # weighted mean of each parameter
        θbar = altar.vector(shape=parameters)
        for j in range(parameters):
            θbar[j] = θ.getColumn(j).mean(weights=w)

        # weighted outer-product sum: Σ += w_i θ_i θ_i^T
        Σ = altar.matrix(shape=(parameters, parameters)).zero()
        for i in range(samples):
            altar.blas.dsyr(Σ.lowerTriangular, w[i], θ.getRow(i), Σ)
        # subtract θ̄ θ̄^T
        altar.blas.dsyr(Σ.lowerTriangular, -1, θbar, Σ)

        # fill the upper triangle
        for i in range(parameters):
            for j in range(i):
                Σ[j, i] = Σ[i, j]

        # condition the covariance matrix if requested
        if self.check_positive_definiteness:
            self.condition_covariance(Σ=Σ)

        return Σ

    def condition_covariance(self, Σ):
        """
        Ensure Σ is symmetric positive definite by lifting small/negative eigenvalues
        """
        altar.libaltar.matrix_condition(Σ.data, self.min_eigenvalue_ratio)
        return Σ

    @altar.export
    def record(self, archiver):
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
    info = None
    rng = None
    uninormal = None

    # owned proposal covariance and its Cholesky factor
    _sigma = None          # the unscaled proposal covariance Σ
    _sigma_chol = None     # Cholesky factor of (scaling^2 * Σ)
    _sigma_is_fixed = False  # True if sigma was set externally via set_sigma()

    # update tracking
    _anneal_count = 0      # number of annealing steps since the last Σ computation
    _prepared_beta = None  # beta value at which _prepare was last called
    _prepared_scaling = None  # scaling at which _prepare was last called

# end of file
