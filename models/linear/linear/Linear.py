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
import numpy
from importlib import import_module
# the package
import altar
# my base class, for its {psets}/{psets_list} and {dataobs} support
from altar.models.BayesianL2 import BayesianL2


# declaration
class Linear(BayesianL2, family="altar.models.linear"):
    """
    A linear model: data = G theta

    My actual forward-model numerics live in {altar.models.linear.native.Linear.Linear} (cpu)
    or {altar.models.linear.cuda.Linear.Linear}; see {_makeImpl} for how one gets picked, the
    same shape {altar.distributions.Base}/{altar.models.Base} already use.
    """


    # user configurable state
    parameters = altar.properties.int(default=None)
    parameters.doc = "the number of parameters in the model"

    # {psets_list}/{psets} and {dataobs} (observations, data covariance, norm) are inherited
    # from {BayesianL2}

    # the name of the test case
    case = altar.properties.path(default="patch-9")
    case.doc = "the directory with the input files"

    # the Green functions; the rest of the forward model's data (observations, covariance,
    # norm) is handled by {dataobs}
    green = altar.properties.path(default="green.txt")
    green.doc = "the name of the file with the Green functions"

    # protocol obligations
    @altar.export
    def initialize(self, application):
        """
        Initialize the state of the model given a {problem} specification
        """
        # chain up; handles job/rng setup, mounting my input dataspace, {dataobs}
        # initialization (observations, data covariance, norm), and laying out my psets
        super().initialize(application=application)

        # pick my backend implementation, once, and let it load the Green functions
        self._impl = self._makeImpl()
        self._impl.initialize(model=self, application=application)

        # grab a channel
        channel = self.debug
        channel.line("run info:")
        # show me the model
        channel.line(f" -- model: {self}")
        # the model state
        channel.line(f" -- model state:")
        channel.line(f"    parameters: {self.parameters}")
        channel.line(f"    observations: {self.observations}")
        # the test case name
        channel.line(f" -- case: {self.case}")
        # the contents of the data filesystem
        channel.line(f" -- contents of '{self.case}':")
        channel.line("\n".join(self.ifs.dump(indent=2)))
        # the loaded data
        channel.line(f" -- inputs in memory:")
        channel.line(f"    green functions: shape={self._impl.G.shape}")
        # flush
        channel.log()

        # all done
        return self


    def _makeImpl(self):
        """
        Build my backend implementation: a same-named class in {native} (the cpu default) or
        {cuda}, picked once, here, based on {altar.backends.active()}
        """
        backend = "cuda" if altar.backends.active() == "cuda" else "native"
        module = import_module(f"altar.models.linear.{backend}.Linear")
        return getattr(module, "Linear")()


    def forward_model_batched(self, theta, prediction, batch=None):
        """
        Fill {prediction}, shape (samples x observations), with the residual G·θ - d for
        each sample in {theta}
        """
        return self._impl.forward_model_batched(
            model=self, theta=theta, prediction=prediction, batch=batch)


    def forward_model(self, theta, green=None, prediction=None, observation=None):
        """
        Linear forward model prediction = G * theta for a single sample, optionally
        subtracting {observation} to get the residual instead
        """
        return self._impl.forward_model(
            model=self, theta=theta, green=green, prediction=prediction, observation=observation)


    def conjugate_posterior(self, mean, variance):
        """
        The posterior N(m*, C*) under the conjugate prior N(mean, diag(variance)), with
        C* = (C_m^-1 + G^T C^-1 G)^-1 and m* = C* (C_m^-1 m + G^T C^-1 d) for the covariance C in
        effect, and the evidence of that model, log p_conj(d) (Minson, 2024, eq. A13)
        """
        G = numpy.asarray(self._impl.green(), dtype=float)
        d = numpy.asarray(self.dataobs.observed(), dtype=float)
        observations = d.size
        covariance = self.dataobs.covariance()
        # whiten G and d by the covariance, C = L L^T
        if isinstance(covariance, float):
            Gw, dw = G / numpy.sqrt(covariance), d / numpy.sqrt(covariance)
            logdet = observations * numpy.log(covariance)
        else:
            L = numpy.linalg.cholesky(covariance)
            Gw, dw = numpy.linalg.solve(L, G), numpy.linalg.solve(L, d)
            logdet = 2 * numpy.log(numpy.diag(L)).sum()
        precision = numpy.diag(1 / variance) + Gw.T @ Gw
        cstar = numpy.linalg.inv(precision)
        cstar = (cstar + cstar.T) / 2
        mstar = cstar @ (mean / variance + Gw.T @ dw)
        _, logdet_star = numpy.linalg.slogdet(cstar)
        log_evidence = 0.5 * (logdet_star - observations * numpy.log(2 * numpy.pi) - logdet
                              - numpy.log(variance).sum()
                              - (dw @ dw + (mean * mean / variance).sum() - mstar @ precision @ mstar))
        return mstar, cstar, float(log_evidence)


    @altar.export
    def forward_problem(self, application, theta):
        """
        The raw predicted data G·θ for each row of {theta}; see {altar.models.Model}
        """
        return {"data": numpy.asarray(theta, dtype=float) @ self._impl.green().T}


    def covariance_updated(self):
        """
        Redo whatever depends on the data covariance, once my implementation exists
        """
        if self._impl is not None:
            self._impl.covariance_updated(model=self)
        return self


    @altar.export
    def gradient(self, controller, step, batch=None):
        """
        Fill {step.grad_prior} and {step.grad_data} with the gradients of the log prior and
        log data likelihood with respect to {step.theta}, for use by gradient-based samplers
        (e.g. SGLD)
        """
        return self._impl.gradient(model=self, controller=controller, step=step, batch=batch)


    # private data
    _impl = None # my backend implementation, chosen once, in {initialize}


# end of file
