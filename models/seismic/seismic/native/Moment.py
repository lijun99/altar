# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# externals
import math
import numpy
# the package
import altar
# my base class
from altar.distributions.native.Uniform import Uniform


# the declaration
class Moment(Uniform):
    """
    The cpu implementation of the moment magnitude prior; see {altar.models.seismic.Moment}
    """


    def initialize(self, rng, application=None):
        """
        The uniform setup, plus the per-patch shear modulus times area
        """
        super().initialize(rng=rng, application=application)
        self.rng = rng.rng
        self.mu_area = mu_area(distribution=self, application=application)
        return self


    def initialize_sample(self, theta, batch=None):
        """
        Fill my portion of {theta} with slips drawn from a gaussian Mw spread by a flat dirichlet
        """
        θ = numpy.asarray(self.restrict(theta=theta))
        θ[:, :] = draw(distribution=self, samples=θ.shape[0], rng=self.rng)
        return self


    def eval_prior(self, theta, likelihood, batch=None):
        """
        Add the uniform log-density, plus the moment constraint if enabled, into {likelihood}
        """
        super().eval_prior(theta=theta, likelihood=likelihood, batch=batch)
        if self.moment_constraint:
            θ = numpy.asarray(self.restrict(theta=theta))
            penalty, _ = moment_penalty(distribution=self, theta=θ)
            L = numpy.asarray(likelihood)
            L[:θ.shape[0]] += penalty
        return self


    def prior_gradient(self, theta, gradient, batch=None):
        r"""
        Fill my portion of {gradient} with d\log P/d\theta: the moment constraint's gradient (the
        uniform part is flat), chained into sampling space when reparameterized
        """
        θ = self.restrict(theta=theta)
        g = self.restrict(theta=gradient)
        G = numpy.asarray(g)
        if self.moment_constraint:
            _, G[:, :] = moment_penalty(distribution=self, theta=numpy.asarray(θ))
        else:
            G[:, :] = 0.0
        if self.reparameterize:
            self.transform.chain_gradient(theta=θ, gradient=g, batch=batch)
        return self


    # private data, set by the shim before {initialize} runs
    area = None
    area_patch_file = None
    Mu = None
    Mw_mean = None
    Mw_sigma = None
    slip_sign = None
    moment_constraint = None
    moment_constraint_factor = None
    # set by {initialize}
    mu_area = None
    rng = None


# helpers shared with the cuda implementation
def mu_area(distribution, application=None):
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


def per_patch(values, patches, name):
    """
    Broadcast a one-value {values} to all {patches}, or check it has one value per patch
    """
    values = numpy.asarray(values, dtype=float).ravel()
    if values.size == 1:
        return numpy.full(patches, values[0])
    if values.size != patches:
        raise ValueError(f"moment: {name} has {values.size} values for {patches} patches")
    return values


def draw(distribution, samples, rng):
    """
    Slips with a gaussian Mw spread over the patches by a flat dirichlet, rejecting any sample
    with a slip outside the support
    """
    patches = distribution.parameters
    low, high = distribution.support
    Mw = altar.pdf.gaussian(mean=distribution.Mw_mean, sigma=distribution.Mw_sigma, rng=rng)
    dirichlet = altar.pdf.dirichlet(alpha=altar.vector(shape=patches).fill(1), rng=rng)
    x = altar.vector(shape=patches)
    sign = -1.0 if distribution.slip_sign == "negative" else 1.0
    θ = numpy.empty((samples, patches))
    for sample in range(samples):
        while True:
            # potency M0/Mu in GPa km^2 m, hence the -15
            potency = sign * 10 ** (1.5 * Mw.sample() + 9.1 - 15)
            dirichlet.vector(vector=x)
            slips = potency * numpy.asarray(x) / distribution.mu_area
            if numpy.all((slips > low) & (slips < high)):
                break
        θ[sample] = slips
    return θ


def moment_penalty(distribution, theta):
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
