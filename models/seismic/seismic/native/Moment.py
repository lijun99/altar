# -*- python -*-
# -*- coding: utf-8 -*-
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
# my base class
from altar.distributions.native.Uniform import Uniform

if typing.TYPE_CHECKING:
    from altar.shells.Application import Application
    from altar.simulations.NumpyRNG import NumpyRNG


# the declaration
class Moment(Uniform):
    """
    The cpu implementation of the moment magnitude prior; see {altar.models.seismic.Moment}
    """


    def initialize(self, rng: NumpyRNG, application: Application | None = None) -> typing.Self:
        """
        The uniform setup, plus the per-patch shear modulus times area
        """
        super().initialize(rng=rng, application=application)
        self.mu_area = mu_area(distribution=self, application=application)
        return self


    def initialize_sample(self, theta: numpy.ndarray, batch: int | None = None) -> typing.Self:
        """
        Fill my portion of {theta} with slips drawn from a gaussian Mw spread by a flat dirichlet
        """
        θ = self.restrict(theta=theta)
        θ[...] = draw(distribution=self, samples=θ.shape[0], rng=self.rng)
        return self


    def eval_prior(self, theta: numpy.ndarray, likelihood: numpy.ndarray,
                   batch: int | None = None) -> typing.Self:
        """
        Add the uniform log-density, plus the moment constraint if enabled, into {likelihood}
        """
        super().eval_prior(theta=theta, likelihood=likelihood, batch=batch)
        if self.moment_constraint:
            θ = self.restrict(theta=theta)
            penalty, _ = moment_penalty(distribution=self, theta=θ)
            likelihood[:θ.shape[0]] += penalty
        return self


    def prior_gradient(self, theta: numpy.ndarray, gradient: numpy.ndarray,
                       batch: int | None = None) -> typing.Self:
        r"""
        Fill my portion of {gradient} with d\log P/d\theta: the moment constraint's gradient (the
        uniform part is flat), chained into sampling space when reparameterized
        """
        θ = self.restrict(theta=theta)
        g = self.restrict(theta=gradient)
        if self.moment_constraint:
            _, g[...] = moment_penalty(distribution=self, theta=θ)
        else:
            g[...] = 0.0
        if self.reparameterize:
            self.transform.chain_gradient(theta=θ, gradient=g, batch=batch)
        return self


    # private data, set by the shim before {initialize} runs
    area: typing.Any = None
    area_patch_file: str | None = None
    Mu: typing.Any = None
    Mw_mean: float
    Mw_sigma: float
    slip_sign: str
    moment_constraint: bool = False
    moment_constraint_factor: float
    # set by {initialize}
    mu_area: numpy.ndarray


# helpers shared with the cuda implementation
def mu_area(distribution: typing.Any, application: Application | None = None) -> numpy.ndarray:
    """
    The per-patch shear modulus times area, from {Mu} and {area} or {area_patch_file}
    """
    patches = distribution.parameters
    area = per_patch(distribution.area, patches, "area")
    if distribution.area_patch_file is not None:
        path = str(distribution.area_patch_file)
        if application is not None:
            path = application.pfs["inputs"][path].uri.path
        area = numpy.loadtxt(path, dtype=float).reshape(patches)
    return per_patch(distribution.Mu, patches, "Mu") * area


def per_patch(values: typing.Any, patches: int, name: str) -> numpy.ndarray:
    """
    Broadcast a one-value {values} to all {patches}, or check it has one value per patch
    """
    values = numpy.asarray(values, dtype=float).ravel()
    if values.size == 1:
        return numpy.full(patches, values[0])
    if values.size != patches:
        raise ValueError(f"moment: {name} has {values.size} values for {patches} patches")
    return values


def draw(distribution: typing.Any, samples: int, rng: numpy.random.Generator) -> numpy.ndarray:
    """
    Slips with a gaussian Mw spread over the patches by a flat dirichlet, redrawing any sample
    with a slip outside the support
    """
    patches = distribution.parameters
    low, high = distribution.support
    sign = -1.0 if distribution.slip_sign == "negative" else 1.0
    θ = numpy.empty((samples, patches))
    pending = numpy.arange(samples)
    while pending.size:
        n = pending.size
        # potency M0/Mu in GPa km^2 m, hence the -15
        potency = sign * 10 ** (1.5 * rng.normal(distribution.Mw_mean, distribution.Mw_sigma, size=n)
                                + 9.1 - 15)
        slips = potency[:, None] * rng.dirichlet(numpy.ones(patches), size=n) / distribution.mu_area
        θ[pending] = slips
        pending = pending[~((slips > low) & (slips < high)).all(axis=1)]
    return θ


def moment_penalty(distribution: typing.Any,
                   theta: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
    """
    The moment constraint's log-density -f (Mw - mean)^2 / (2 sigma^2) per sample, and its
    gradient with respect to {theta}; Mw = (log10|M0| + 5.9)/1.5 with Mu in GPa, A in km^2
    """
    f = distribution.moment_constraint_factor
    s2 = distribution.Mw_sigma ** 2
    M0 = theta @ distribution.mu_area
    dMw = (numpy.log10(numpy.abs(M0)) + 5.9) / 1.5 - distribution.Mw_mean
    penalty = -f * dMw * dMw / (2 * s2)
    gradient = (-f * dMw / s2 / (1.5 * math.log(10) * M0))[:, None] * distribution.mu_area
    return penalty, gradient


# end of file
