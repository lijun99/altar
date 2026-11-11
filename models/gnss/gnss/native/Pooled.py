# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#


# externals
import numpy
# my base class
from altar.distributions.native.Base import Base as base
# the geometry of the pools
from .. import pooling


class Pooled(base):
    """
    The cpu implementation of the spatially pooled sparsity prior, with numpy and a dense matrix
    of neighbor weights; see {altar.models.gnss.Pooled}
    """


    def initialize(self, rng, application=None):
        """
        Load the geometry of my group from the case file
        """
        super().initialize(rng=rng, application=application)
        path = resolve(self.design_file, application)
        self.geometry = pooling.load(path, self.parameters, self.length, self.radius,
                                     self.pooling)
        info = getattr(application, "info", None)
        if info is not None:
            info.log(f"pooled prior, {self.shape}, scale {self.scale:g}: "
                     f"{self.geometry.describe()}")
        return self


    def draw(self, shape):
        """
        Independent Laplace draws of my scale
        """
        return self.rng.laplace(0.0, self.scale, size=shape)


    def verify(self, theta, mask, batch=None):
        """
        Unbounded: every sample is valid
        """
        return mask


    def terms(self, theta):
        """
        The log prior of each sample, the samples in station, component, column layout, and
        phi'(r) / r of each pooled energy
        """
        g = self.geometry
        θ = numpy.asarray(self.restrict(theta=theta), dtype=float)
        S = θ.shape[0]
        blocks = θ.reshape(S, g.blocks, g.width)
        X = numpy.zeros((S, len(g.stations), len(g.components), g.width))
        k, c = numpy.nonzero(g.block >= 0)
        X[:, k, c, :] = blocks[:, g.block[k, c], :]
        Q = numpy.stack([(X[:, :, p, :] ** 2).sum(axis=2) for p in g.pools], axis=2)
        E = numpy.einsum("sk,nkqj->nsqj", g.W, Q)
        r = numpy.sqrt(E + self.epsilon ** 2)
        if self.shape == "laplace":
            phi, f = r / self.scale, 1 / (self.scale * r)
        else:
            phi = self.strength * numpy.log1p(r / self.scale)
            f = self.strength / ((self.scale + r) * r)
        return -phi.sum(axis=(1, 2, 3)), X, f


    def eval_prior(self, theta, likelihood, batch=None):
        """
        Add the log prior of each sample to {likelihood}
        """
        logp, _, _ = self.terms(theta)
        likelihood[:logp.size] += logp
        return self


    def log_density(self, x):
        """
        The log prior of each row of {x}, my parameters only
        """
        x = numpy.atleast_2d(numpy.asarray(x, dtype=float))
        full = numpy.zeros((x.shape[0], self.offset + self.parameters))
        full[:, self.offset:] = x
        return self.terms(full)[0]


    def prior_gradient(self, theta, gradient, batch=None):
        """
        Fill my columns of {gradient} with d log P / d theta = -theta (W^T f) per pool
        """
        g = self.geometry
        _, X, f = self.terms(theta)
        F = numpy.einsum("sk,nsqj->nkqj", g.W, f)
        grad = -X * F[:, :, g.pool_of, :]
        out = self.restrict(theta=gradient).reshape(X.shape[0], g.blocks, g.width)
        k, c = numpy.nonzero(g.block >= 0)
        out[:, g.block[k, c], :] = grad[:, k, c, :]
        return self


    # private data, set by the shim before {initialize} runs
    design_file = None
    scale = None
    shape = None
    strength = None
    epsilon = None
    length = None
    radius = None
    pooling = None
    geometry = None


def resolve(path, application=None):
    """
    The case file: relative to the current directory, or to the model's input directory
    """
    import os
    path = str(path)
    if not os.path.exists(path) and application is not None:
        path = application.pfs["inputs"][path].uri.path
    return path


# end of file
