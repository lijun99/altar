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
from importlib.util import find_spec
import numpy
# the package
import altar
from altar.models.BayesianL2 import BayesianL2

if typing.TYPE_CHECKING:
    from altar.bayesian.controllers.Annealer import Annealer
    from altar.bayesian.states.BayesianState import BayesianState
    from altar.shells.Application import Application
    from altar.arrays import Array

# declaration
class Linear(BayesianL2, family="altar.models.regression.linear"):
    """
    Linear regression, y = slope * x + intercept, with the parameter sets {slope} and
    {intercept}; the y are the observed data, {dataobs}, and the x are read from {x_file}

    On the cpu, the forward model and the gradient are numpy's; on the gpu, they are kernels in
    CUDA C++ compiled at run time by NVRTC ({NVRTC}), or in python with cuTile ({CuTile})
    """

    # user configurable state
    x_file = altar.properties.path(default="x.txt")
    x_file.doc = "the input file with the x of each observation"

    # the forward model computes the predictions, and {dataobs} the residuals
    return_residual = altar.properties.bool(default=False)
    return_residual.doc = "the forward model returns residual(True) or prediction(False)"

    kernels = altar.properties.str(default="nvrtc")
    kernels.doc = "the gpu kernels: CUDA C++ compiled at run time (nvrtc), or python (cutile)"
    kernels.validators = altar.constraints.isMember("nvrtc", "cutile")


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
        # on the gpu, pick my kernels
        if altar.backends.active() == "cuda":
            self._impl = self._makeImpl()
            self._impl.initialize(model=self)
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


    def forward_model_batched(self, theta: Array, prediction: Array,
                              batch: int | None = None) -> typing.Self:
        """
        Fill the first {batch} rows of {prediction} with the predicted y of {theta}: with my
        kernels on the gpu, one sample at a time with {forward_model} on the cpu
        """
        if self._impl is None:
            return super().forward_model_batched(theta=theta, prediction=prediction, batch=batch)
        batch = theta.shape[0] if batch is None else batch
        self._impl.forward_model_batched(theta=theta, prediction=prediction, batch=batch)
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
        # on the gpu, my kernels do it
        if self._impl is not None:
            return self.gradient_cuda(step=step, batch=batch)
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
    def gradient_cuda(self, step: BayesianState, batch: int | None = None) -> typing.Self:
        """
        The gradients on the gpu, in {step.prior_gradient} and {step.data_gradient}, as the cuda
        states name them
        """
        for name in self.psets_list:
            self.psets[name].prior_gradient(
                theta=step.theta, gradient=step.prior_gradient, batch=batch)
        # the data and the inverse variance of each observation, uploaded once
        if self._weight is None:
            cd_inv = self.dataobs.cd_inv
            if not isinstance(cd_inv, float):
                self.error.log("the regression gpu gradient needs a constant covariance, cd_std")
                raise SystemExit(1)
            weight = numpy.full(self.observations, cd_inv**2)
            if self.dataobs.mask is not None:
                weight[~self.dataobs.mask] = 0
            self._data = altar.cuda.vector(source=self.dataobs.observed(), dtype=self.precision)
            self._weight = altar.cuda.vector(source=weight, dtype=self.precision)
        batch = step.theta.shape[0] if batch is None else batch
        self._impl.gradient(theta=step.theta, data=self._data, weight=self._weight,
                            gradient=step.data_gradient, batch=batch)
        # all done
        return self


    def _makeImpl(self):
        """
        Build my gpu implementation, with the {kernels} of my choice
        """
        package, pip = (("cuda.tile", "'cuda-tile[tileiras]'") if self.kernels == "cutile"
                        else ("cuda.core", "cuda-core"))
        if find_spec("cuda") is None or find_spec(package) is None:
            self.error.log(f"the {self.kernels} kernels need {package}, pip install {pip}; "
                           f"or run on the cpu, with job.gpus = 0")
            raise SystemExit(1)
        if self.kernels == "cutile":
            from .CuTile import CuTile as strategy
        else:
            from .NVRTC import NVRTC as strategy
        return strategy()


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
    _impl = None # my gpu implementation, if i run on the gpu
    _data = None # the observed y, and the inverse variance of each, on the gpu
    _weight = None

# end of file
