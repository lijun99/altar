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
    Implementation of Gaussian proposal kernel for MCMC sampling.
    """

    # user configurable state
    check_positive_definiteness = altar.properties.bool(default=True)
    check_positive_definiteness.doc = 'whether to check the positive definiteness of Σ matrix and condition it accordingly'

    min_eigenvalue_ratio = altar.properties.float(default=0.001)
    min_eigenvalue_ratio.doc = 'the desired minimal eigenvalue of Σ matrix, as a ratio to the max eigenvalue'

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

        # all done
        return self

    @altar.export
    def propose(self, sampler, step, annealer=None):
        """
        Propose a new sample set using a Gaussian random walk
        """
        # update the proposal state if needed
        if self._needs_prepare(step=step, scaling=sampler.scaling):
            self._prepare(sampler=sampler, step=step, annealer=annealer)

        # generate the displaced samples
        return self._displace(sample=step.theta)


    # implementation details
    def _prepare(self, sampler, step, annealer):
        """
        Compute the covariance and prepare the sampling pdf
        """
        dispatcher = annealer.dispatcher if annealer is not None else None
        if dispatcher is not None:
            dispatcher.notify(event=dispatcher.prepareSamplingPDFStart, controller=annealer)

        # propagate legacy scheduler knobs, if any
        if annealer is not None:
            scheduler = annealer.scheduler
            if hasattr(scheduler, "check_positive_definiteness"):
                self.check_positive_definiteness = scheduler.check_positive_definiteness
            if hasattr(scheduler, "min_eigenvalue_ratio"):
                self.min_eigenvalue_ratio = scheduler.min_eigenvalue_ratio

        # determine weights
        samples = step.samples
        weights = None
        if annealer is not None:
            weights = getattr(annealer.scheduler, "w", None)
        if weights is None:
            weights = altar.vector(shape=samples).zero()
            value = 1.0 / samples
            for i in range(samples):
                weights[i] = value

        # compute and store the covariance
        Σ = self.computeCovariance(step=step, w=weights)
        step.sigma.copy(Σ)

        # scale and decompose it
        Σ = step.sigma.clone()
        Σ *= sampler.scaling**2
        self.sigma_chol = altar.lapack.cholesky_decomposition(Σ)

        # cache preparation state
        self._prepared_step_id = id(step)
        self._prepared_scaling = sampler.scaling

        if dispatcher is not None:
            dispatcher.notify(event=dispatcher.prepareSamplingPDFFinish, controller=annealer)

        # all done
        return self


    def _needs_prepare(self, step, scaling):
        """
        Decide whether we need to refresh the cached proposal state
        """
        if self.sigma_chol is None:
            return True
        if self._prepared_step_id != id(step):
            return True
        if self._prepared_scaling != scaling:
            return True
        return False


    def _displace(self, sample):
        """
        Construct a set of displacement vectors for the random walk
        """
        # get my decomposed covariance
        Σ_chol = self.sigma_chol

        # build a set of random displacement vectors; note that, for convenience, this starts
        # out as (parameters x samples), i.e. the transpose of what we need
        δT = altar.matrix(shape=tuple(reversed(sample.shape))).random(pdf=self.uninormal)
        # multiply the displacement vectors by the decomposed covariance
        δT = altar.blas.dtrmm(
            Σ_chol.sideLeft, Σ_chol.lowerTriangular, Σ_chol.opNoTrans, Σ_chol.nonUnitDiagonal,
            1, Σ_chol, δT)

        # allocate the transpose
        δ = altar.matrix(shape=sample.shape)
        # fill it
        δT.transpose(δ)
        # offset it by the original sample
        δ += sample
        # and return it
        return δ

    @altar.export
    def computeCovariance(self, step, w):
        r"""
        Compute the parameter covariance Σ of the sample in {step}

          Σ = c_m^2 \sum_{i \in samples} \tilde{w}_{i} θ_i θ_i^T} - \bar{θ} \bar{θ}^Τ

        where

          \bar{θ} = \sum_{i \in samples} \tilde{w}_{i} θ_{i}

        The covariance Σ gets used to build a proposal pdf for the posterior
        """
        # unpack what i need
        θ = step.theta # the current sample set
        # extract the number of samples and number of parameters
        samples = step.samples
        parameters = step.parameters

        # initialize the covariance matrix
        Σ = altar.matrix(shape=(parameters, parameters)).zero()

        # check the geometries
        assert w.shape == samples
        assert θ.shape == (samples, parameters)
        assert Σ.shape == (parameters, parameters)

        # calculate the weighted mean of every parameter across all samples
        θbar = altar.vector(shape=parameters)
        # for each parameter
        for j in range(parameters):
            # the jth column in θ has the value of this parameter in the various samples
            θbar[j] = θ.getColumn(j).mean(weights=w)
        # start filling out Σ
        for i in range(samples):
            # get the sample
            sample = θ.getRow(i)
            # form Σ += w[i] sample sample^T
            altar.blas.dsyr(Σ.lowerTriangular, w[i], sample, Σ)
        # subtract θbar θbar^T
        altar.blas.dsyr(Σ.lowerTriangular, -1, θbar, Σ)
        # fill the upper triangle
        for i in range(parameters):
            for j in range(i):
                Σ[j,i] = Σ[i,j]

        # condition the covariance matrix
        if self.check_positive_definiteness:
            self.conditionCovariance(Σ=Σ)

        # all done
        return Σ

    def conditionCovariance(self, Σ):
        """
        Make sure the covariance matrix Σ is symmetric and positive definite
        """
        # replaces negative or small eigenvalues with min_eigenvalue_ratio*max_eigenvalue
        altar.libaltar.matrix_condition(Σ.data, self.min_eigenvalue_ratio)
        # all done
        return Σ

    # private data
    info = None
    rng = None
    uninormal = None
    sigma_chol = None
    _prepared_step_id = None
    _prepared_scaling = None

# end of file
