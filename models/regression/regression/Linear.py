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
# the package
import altar
from altar.models.BayesianL2 import BayesianL2

if typing.TYPE_CHECKING:
    from altar.bayesian.controllers.Annealer import Annealer
    from altar.bayesian.states.BayesianState import BayesianState
    from altar.shells.Application import Application

# declaration
class Linear(BayesianL2, family="altar.models.regression.linear"):
    """
    Linear regression, y = slope * x + intercept, with the parameter sets {slope} and
    {intercept}; the y are the observed data, {dataobs}, and the x are read from {x_file}
    """

    # user configurable state
    x_file = altar.properties.path(default="x.txt")
    x_file.doc = "the input file with the x of each observation"

    # the forward model computes the predictions, and {dataobs} the residuals
    return_residual = altar.properties.bool(default=False)
    return_residual.doc = "the forward model returns residual(True) or prediction(False)"


    # protocol obligations
    @altar.export
    def initialize(self, application: Application) -> typing.Self:
        """
        Initialize the state of the model given an {application} context
        """
        # chain up; mounts my input dataspace, loads the observations and lays out my psets
        super().initialize(application=application)
        # load the x
        self.x = self.io.load(filename=self.x_file, shape=self.observations)
        # find my parameters in a sample
        for name in ("slope", "intercept"):
            if name not in self.psets_list or self.psets[name].count != 1:
                channel = self.error
                channel.log(f"the regression model needs a parameter set '{name}' with count=1")
                raise SystemExit(1)
        self.slopeIdx = self.psets["slope"].offset
        self.interceptIdx = self.psets["intercept"].offset
        # all done
        return self


    def forward_model(self, theta: numpy.ndarray, prediction: numpy.ndarray) -> typing.Self:
        """
        Fill {prediction} with the predicted y of a single sample {theta}
        """
        # grab my parameters
        slope = theta[self.slopeIdx]
        intercept = theta[self.interceptIdx]
        # and predict the data
        prediction[:] = slope * self.x + intercept
        # all done
        return self


    @altar.export
    def forward_problem(self, application: Application, theta: numpy.ndarray) -> dict:
        """
        The predicted y for each row of {theta}; see {altar.models.Model}
        """
        return {"data": self.predict(theta=numpy.asarray(theta, dtype=float))}


    @altar.export
    def gradient(self, controller: Annealer, step: BayesianState,
                 batch: int | None = None) -> typing.Self:
        """
        Fill {step.grad_prior} and {step.grad_data} with the gradients of the log prior and of
        the log data likelihood with respect to {step.theta}, for the gradient-based samplers
        """
        # these samplers need unbounded priors, or reparameterized ones
        if not self.checked_unbounded_priors:
            self.verify_unbounded_priors()
        # the prior gradient, from my parameter sets
        for name in self.psets_list:
            self.psets[name].prior_gradient(theta=step.theta, gradient=step.grad_prior)
        # the residuals r = prediction - d, (samples x observations)
        batch = step.theta.shape[0] if batch is None else batch
        r = self.predict(theta=step.theta[:batch]) - self.dataobs.observed()
        # weighted by the inverse data covariance, C^{-1} r, with C^{-1} = L L^T
        cd_inv = self.dataobs.cd_inv
        if isinstance(cd_inv, float):
            w = r * cd_inv**2
            # skip the masked observations
            if self.dataobs.mask is not None:
                w[:, ~self.dataobs.mask] = 0
        else:
            w = r @ cd_inv @ cd_inv.T
        # the log likelihood is -r^T C^{-1} r / 2, so its gradient is -(dr/dθ)^T C^{-1} r
        step.grad_data[:batch, self.slopeIdx] = -w @ self.x
        step.grad_data[:batch, self.interceptIdx] = -w.sum(axis=1)
        # all done
        return self


    # implementation details
    def predict(self, theta: numpy.ndarray) -> numpy.ndarray:
        """
        The predicted y for each row of the numpy array {theta}, (samples x observations)
        """
        θ = numpy.atleast_2d(theta)
        return θ[:, self.slopeIdx, None] * self.x + θ[:, self.interceptIdx, None]


    # private data
    x: numpy.ndarray # the x of each observation
    slopeIdx: int # where my parameters are in a sample
    interceptIdx: int

# end of file
