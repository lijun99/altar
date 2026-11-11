# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#


# externals
import numpy


# declaration
class Trajectory:
    """
    The cpu numerics of the gnss trajectory model, with numpy: per group, one matrix product
    for all the blocks of all the samples if the group is shared, an einsum if not
    """


    def initialize(self, model, application):
        """
        Keep my design, the offsets of its groups, and the whitened observations
        """
        self.design, self.offsets = model.design, model.offsets
        self.data = numpy.asarray(model.dataobs.observed(), dtype=float)
        return self


    def forward_model_batched(self, model, theta, prediction, batch=None):
        """
        The whitened residuals of the first {batch} rows of {theta}
        """
        batch = theta.shape[0] if batch is None else batch
        prediction[:batch] = self.design.predict(theta[:batch], self.offsets) - self.data
        return self


    def gradient(self, model, controller, step, batch=None):
        """
        The prior gradient from my parameter sets, and the data gradient -J^T r
        """
        θ = step.theta
        batch = θ.shape[0] if batch is None else batch
        for name in model.psets_list:
            model.psets[name].prior_gradient(theta=θ, gradient=step.grad_prior)
        r = self.design.predict(θ[:batch], self.offsets) - self.data
        self.design.adjoint(-r, self.offsets, step.grad_data[:batch])
        return self


    # private data
    design = None
    offsets = None
    data = None


# end of file
