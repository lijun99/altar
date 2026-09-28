# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# externals
import numpy
# the package
import altar
import altar.cuda
# my base class
from altar.distributions.cuda.Uniform import Uniform
# the numerics shared with the cpu implementation
from ..native.Moment import mu_area, draw


# the declaration
class Moment(Uniform):
    """
    The cuda implementation of the moment magnitude prior; see {altar.models.seismic.Moment}
    """


    def initialize(self, rng, application=None):
        """
        The uniform setup, plus the per-patch shear modulus times area, on the device
        """
        super().initialize(rng=rng, application=application)
        # the moment kernels
        from altar.models.seismic.ext import cudaseismic
        self.libcudaseismic = cudaseismic
        # {rng} is unused on cuda; initial samples are drawn once, on the host
        self.rng = application.rng.rng
        self.mu_area = mu_area(distribution=self, application=application)
        self.g_mu_area = altar.cuda.vector(shape=self.parameters, dtype=self.precision)
        numpy.asarray(self.g_mu_area)[:] = self.mu_area
        return self


    def initialize_sample(self, theta, batch=None):
        """
        Fill my portion of {theta} with slips drawn from a gaussian Mw spread by a flat dirichlet;
        a one-off host-side draw into managed memory
        """
        θ = numpy.asarray(self._grid(theta))
        θ[:, self.idx_begin:self.idx_end] = draw(distribution=self, samples=θ.shape[0], rng=self.rng)
        return self


    def eval_prior(self, theta, likelihood, batch=None):
        """
        Add the uniform log-density, plus the moment constraint if enabled, into {likelihood}
        """
        super().eval_prior(theta=theta, likelihood=likelihood, batch=batch)
        if self.moment_constraint:
            self.libcudaseismic.cudaMoment_logpdf(
                self._grid(theta), self._grid(likelihood), self.idx_begin, self.idx_end,
                self.Mw_mean, self.Mw_sigma, self._grid(self.g_mu_area), self.moment_constraint_factor)
        return self


    def prior_gradient(self, theta, gradient, batch=None):
        r"""
        Fill my portion of {gradient} with d\log P/d\theta: the moment constraint's gradient (the
        uniform part is flat), chained into sampling space when reparameterized
        """
        if self.moment_constraint:
            self.libcudaseismic.cudaMoment_logpdfgradient(
                self._grid(theta), self._grid(gradient), self.idx_begin, self.idx_end,
                self.Mw_mean, self.Mw_sigma, self._grid(self.g_mu_area), self.moment_constraint_factor)
            if self.reparameterize:
                self.transform.chain_gradient(theta=theta, gradient=gradient, batch=batch)
        elif self.reparameterize:
            self.transform.jacobian_gradient(theta=theta, gradient=gradient, batch=batch)
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
    g_mu_area = None
    rng = None
    libcudaseismic = None


# end of file
