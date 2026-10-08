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
import typing
import numpy

if typing.TYPE_CHECKING:
    from altar.bayesian.controllers.Annealer import Annealer
    from altar.bayesian.states.BayesianState import BayesianState
    from altar.shells.Application import Application
    from ..Linear import Linear as Model


# declaration
class Linear:
    """
    The cpu implementation of the linear forward model: data = G theta

    My configuration ({green}) is read directly off the shim's {model}, once, in
    {initialize}, since it's only ever needed there.
    """


    def initialize(self, model: Model, application: Application) -> typing.Self:
        """
        Load the Green functions and the observed data
        """
        # load my Green functions
        self.G = model.io.load(filename=model.green, shape=(model.observations, model.parameters))
        # the data the residuals are measured against
        self.data = model.dataobs.dataobs
        # all done
        return self


    def forward_model_batched(self, model: Model, theta: numpy.ndarray, prediction: numpy.ndarray,
                              batch: int | None = None) -> typing.Self:
        """
        Fill {prediction}, shape (samples x observations), with the residual G·θ - d for
        each sample in {theta}
        """
        prediction[...] = theta @ self.G.T - self.data
        # all done
        return self


    def forward_model(self, model: Model, theta: numpy.ndarray, green: numpy.ndarray | None = None,
                      prediction: numpy.ndarray | None = None,
                      observation: numpy.ndarray | None = None) -> numpy.ndarray:
        """
        Linear forward model prediction = G * theta for a single sample, optionally
        subtracting {observation} to get the residual instead
        """
        green = self.G if green is None else green
        if prediction is None:
            prediction = numpy.zeros(model.observations)
        prediction[...] = green @ theta
        if observation is not None:
            prediction -= observation
        # all done
        return prediction


    def covariance_updated(self, model: Model) -> typing.Self:
        """
        The observed data may have changed with the covariance; refresh them
        """
        self.data = model.dataobs.dataobs
        return self


    def green(self) -> numpy.ndarray:
        """
        The raw green's functions (observations x parameters)
        """
        return self.G


    def gradient(self, model: Model, controller: Annealer, step: BayesianState,
                 batch: int | None = None) -> typing.Self:
        """
        Fill {step.grad_prior} and {step.grad_data} with the gradients of the log prior and
        log data likelihood with respect to {step.theta}, for use by gradient-based samplers
        (e.g. SGLD)
        """
        # gradient-based samplers have no accept/reject step to catch a proposal that walked
        # outside a bounded prior's support, so they currently only support unbounded priors;
        # check once and cache, since {gradient} is called on every sweep
        # in an ensemble, the ensemble owns the parameter sets and their priors
        if not model.embedded and not model.checked_unbounded_priors:
            model.verify_unbounded_priors()

        # grab the portion of the sample, and of the gradient buffers, that are mine
        θ = model.restrict(theta=step.theta)
        grad_prior = model.restrict(theta=step.grad_prior)
        grad_data = model.restrict(theta=step.grad_data)

        # the prior gradient, in {psets_list} order -- {psets} is a dict and may carry extra
        # entries merged in from other configuration sources
        for name in ([] if model.embedded else model.psets_list):
            model.psets[name].prior_gradient(theta=θ, gradient=grad_prior)

        # the data likelihood gradient: for r = Gθ - d and Cd^{-1} = L L^T, with L the lower
        # Cholesky factor in {dataobs.cd_inv}, grad = -G^T Cd^{-1} r, one row per sample
        r = θ @ self.G.T - self.data
        Cd_inv = model.dataobs.cd_inv
        if isinstance(Cd_inv, float):
            # a constant covariance, {Cd_inv} = 1/sigma, over the valid observations only
            wt = r * (Cd_inv * Cd_inv)
            mask = model.dataobs.mask
            if mask is not None:
                wt[:, ~mask] = 0
        else:
            wt = r @ Cd_inv @ Cd_inv.T
        grad_data[...] = -wt @ self.G

        # all done
        return self


    # private data
    G: numpy.ndarray     # the Green functions, (observations x parameters)
    data: numpy.ndarray  # the observed data the residuals are measured against


# end of file
