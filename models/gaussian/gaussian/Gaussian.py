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
import math
import typing
import numpy
# the package
import altar

if typing.TYPE_CHECKING:
    from altar.bayesian.states.BayesianState import BayesianState
    from altar.shells.Application import Application


# declaration
class Gaussian(altar.models.bayesian, family="altar.models.gaussian"):
    """
    A model that emulates the probability density for a single observation of the model
    parameters. The observation is treated as normally distributed around a given mean, with a
    covariance constructed out of its eigenvalues and a rotation in configuration
    space. Currently, only two dimensional parameter spaces are supported.
    """


    # user configurable state
    parameters = altar.properties.int(default=2)
    parameters.doc = "the number of model degrees of freedom"

    support = altar.properties.array(default=(-1,1))
    support.doc = "the support interval of the prior distribution"

    prep = altar.distributions.distribution()
    prep.doc = "the distribution used to generate the initial sample"

    prior = altar.distributions.distribution()
    prior.doc = "the prior distribution"

    μ = altar.properties.array(default=(0,0))
    μ.doc = 'the location of the central value of the observation'

    λ = altar.properties.array(default=(.01, .005))
    λ.doc = 'the eigenvalues of the covariance matrix'

    φ = altar.properties.dimensional(default=0*altar.units.angle.rad)
    φ.doc = 'the orientation of the covariance semi-major axis'


    # protocol obligations
    @altar.export
    def initialize(self, application: Application) -> typing.Self:
        """
        Initialize the state of the model given a {problem} specification
        """
        # cpu only: my numerics index the host arrays of the samples
        if altar.backends.active() == "cuda":
            application.error.log(
                "the gaussian model has only a cpu implementation; run it with job.gpus = 0")
            raise SystemExit(1)
        # chain up
        super().initialize(application=application)
        # get my random number generator
        rng = self.rng

        # initialize my distributions
        self.prep.initialize(rng=rng, application=application)
        self.prior.initialize(rng=rng, application=application)

        # all done
        return self


    @altar.export
    def initialize_sample(self, step: BayesianState, batch: int | None = None) -> typing.Self:
        """
        Fill {step.θ} with an initial random sample from my prior distribution.
        """
        # grab the portion of the sample that's mine
        θ = self.restrict(theta=step.theta)
        # fill it with random numbers from my initializer
        self.prep.initialize_sample(theta=θ)
        # and return
        return self


    @altar.export
    def eval_prior(self, step: BayesianState, batch: int | None = None) -> typing.Self:
        """
        Fill {step.prior} with the likelihoods of the samples in {step.theta} in the prior
        distribution
        """
        # grab my prior pdf
        pdf = self.prior
        # grab the portion of the sample that's mine
        θ = self.restrict(theta=step.theta)
        # and the storage for the prior likelihoods
        likelihood = step.prior

        # delegate
        pdf.eval_prior(theta=θ, likelihood=likelihood)

        # all done
        return self


    @altar.export
    def data_likelihood(self, step: BayesianState, batch: int | None = None) -> typing.Self:
        """
        Fill {step.data} with the likelihoods of the samples in {step.theta} given the available
        data. This is what is usually referred to as the "forward model"
        """
        # grab the portion of the sample that's mine, as differences from the mean
        δ = self.restrict(theta=step.theta) - self.peak
        # the log-likelihood of the data given each sample, from {δ^T . σ_inv . δ}
        v = numpy.einsum("si,ij,sj->s", δ, self.σ_inv, δ)
        step.data[:δ.shape[0]] += self.normalization - v/2

        # all done
        return self


    @altar.export
    def verify(self, step: BayesianState, mask: numpy.ndarray,
               batch: int | None = None) -> numpy.ndarray:
        """
        Check whether the samples in {step.theta} are consistent with the model requirements and
        update the {mask}, a vector with zeroes for valid samples and non-zero for invalid ones
        """
        # grab the portion of the sample that's mine
        θ = self.restrict(theta=step.theta)
        # grab my prior
        pdf = self.prior
        # ask it to verify my samples
        pdf.verify(theta=θ, mask=mask)
        # all done; return the rejection map
        return mask


    # meta methods
    def __init__(self, **kwds) -> None:
        # chain up
        super().__init__(**kwds)

        # local names for the math functions
        log, π, cos, sin = math.log, math.pi, math.cos, math.sin

        # the number of model parameters
        dof = self.parameters

        # the central value
        peak = numpy.array(self.μ, dtype=float)

        # the trigonometry
        cos_φ = cos(self.φ)
        sin_φ = sin(self.φ)
        # the eigenvalues
        λ0 = self.λ[0]
        λ1 = self.λ[1]
        # the eigenvalue inverses
        λ0_inv = 1/λ0
        λ1_inv = 1/λ1

        # build the inverse of the covariance matrix
        σ_inv = numpy.zeros((dof, dof))
        σ_inv[0,0] = λ0_inv*cos_φ**2 +  λ1_inv*sin_φ**2
        σ_inv[1,1] = λ1_inv*cos_φ**2 +  λ0_inv*sin_φ**2
        σ_inv[0,1] = σ_inv[1,0] = (λ1_inv - λ0_inv) * cos_φ * sin_φ

        # compute its determinant and store it
        σ_lndet = log(λ0 * λ1)

        # attach the characteristics of my pdf
        self.peak = peak
        self.σ_inv = σ_inv

        # the log-normalization
        self.normalization = -.5*(dof*log(2*π) + σ_lndet)

        # all done
        return


    # implementation details
    peak: numpy.ndarray # the location of my central value
    σ_inv: numpy.ndarray # the inverse of my data covariance
    normalization: float = 1 # the normalization factor for my prior distribution


# end of file
