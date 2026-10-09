# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#


# the package
import altar
# my parts: the jax forward model and gradient, next to me, and the linear model for the rest
from JaxModel import JaxModel
from altar.models.linear.Linear import Linear


# declaration
class LinearJax(JaxModel, Linear, family="altar.models.linear.jax"):
    """
    The linear model, data = G theta, with its forward model and gradient in jax
    """


    # my predictions are raw: {dataobs} subtracts the data and applies the covariance
    return_residual = altar.properties.bool(default=False)
    return_residual.doc = "the forward model returns residual(True) or prediction(False)"


    # protocol obligations
    @altar.export
    def initialize(self, application):
        """
        Load the Green functions, as {Linear} does, and keep them for jax
        """
        super().initialize(application=application)
        import jax.numpy as jnp
        # the raw Green functions, in the precision of the chains
        self._jax()
        self._G = jnp.asarray(self._impl.green(), dtype=self.precision)
        return self


    # the forward model of one sample
    def jax_forward(self, theta):
        return self._G @ theta


    # private data
    _G = None # the Green functions, (observations x parameters), as a jax array


# end of file
