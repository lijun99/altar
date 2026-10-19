# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# externals
from __future__ import annotations
import typing
from importlib import import_module
import numpy
# the package
import altar
# my protocol, for the model i wrap
from .Model import Model
# my base class
from .Bayesian import Bayesian

if typing.TYPE_CHECKING:
    from altar.arrays import Array
    from altar.bayesian.controllers.Annealer import Annealer
    from altar.bayesian.states.BayesianState import BayesianState
    from altar.distributions.native.MultivariateGaussian import MultivariateGaussian
    from altar.shells.Application import Application


# declaration
class CrossFade(Bayesian, family="altar.models.crossfade"):
    """
    Cross-fade sampling (Minson, 2024, GJI 239, 1629) of the posterior of {model}

    {model} provides a conjugate prior p_conj(θ) = N(m, C_m), whose posterior
    p_conj(θ|d) = N(m*, C*) is known in closed form, e.g. the linear and the static slip models.
    I sample f_β(θ) ∝ p_conj(θ|d) [p(θ) / p_conj(θ)]^β from β = 0 to 1: my prior slot holds
    log p_conj(θ|d), my data slot log p(θ) - log p_conj(θ), so the data likelihood is never
    evaluated; the samples start from the conjugate posterior, and the evidence is
    p(d) = p_conj(d) q Π_m <w_m>, with q the fraction of the conjugate posterior within the
    support of the prior. Everything else, the parameter sets, the support checks, the
    reparameterization, the forward problem, is {model}'s.

    I run under the {cf_catmip} controller, which seeds the evidence of the annealing with mine,
    and wraps any other model in me.
    """

    # user configurable state
    model = Model()
    model.doc = "the model whose posterior i sample, which provides its conjugate posterior"


    # protocol obligations
    @altar.export
    def initialize(self, application: Application) -> typing.Self:
        """
        Initialize {model}, then its conjugate posterior and my initial samples
        """
        # chain up
        super().initialize(application=application)
        # set up the model i wrap
        self.model = self.model.initialize(application=application)
        # and get ready to sample
        return self.prepare()


    def adopt(self, model: Bayesian) -> typing.Self:
        """
        Wrap {model}, already initialized, and get ready to sample its posterior
        """
        self.model = model
        # borrow its context
        self.job = model.job
        self.rng = model.rng
        self.controller = model.controller
        self.info, self.warning, self.error = model.info, model.warning, model.error
        self.debug, self.firewall = model.debug, model.firewall
        # and get ready to sample
        return self.prepare()


    def prepare(self) -> typing.Self:
        """
        Compute the conjugate prior and posterior of {model}, the evidence of the conjugate
        model, and draw my initial samples from the conjugate posterior
        """
        try:
            return self._prepare()
        except (NotImplementedError, ValueError) as reason:
            self.error.log(f"cross-fade sampling: {reason}")
            raise SystemExit(1)


    def _prepare(self) -> typing.Self:
        """
        The work of {prepare}, which raises when {model} can't be cross-faded
        """
        from .BayesianL2 import BayesianL2
        model = self.model
        if not isinstance(model, BayesianL2):
            raise NotImplementedError(
                f"'{type(model).__name__}' has no conjugate posterior, for cross-fade sampling")
        if model.embedded:
            raise NotImplementedError("cross-fade sampling of a model in an ensemble")
        # my layout and precision are those of the model
        self.parameters = model.parameters
        self.precision = model.precision
        self.cuda = altar.backends.active() == "cuda"
        # the multivariate normal distribution of my backend
        backend = "cuda" if self.cuda else "native"
        MultivariateGaussian = import_module(
            f"altar.distributions.{backend}.MultivariateGaussian").MultivariateGaussian
        # the conjugate prior N(m, C_m) and posterior N(m*, C*), and log p_conj(d)
        mean, variance = model.conjugate_prior()
        mstar, cstar, log_evidence = model.conjugate_posterior(mean=mean, variance=variance)
        self.conjugate_prior = MultivariateGaussian(
            mean=mean, covariance=numpy.diag(variance), precision=self.precision)
        self.conjugate_posterior = MultivariateGaussian(
            mean=mstar, covariance=cstar, precision=self.precision)
        # the initial samples, and the fraction of the conjugate posterior within the support
        coverage = self.draw(rows=model.samples)
        # the evidence of the conjugate model within the support
        self.log_evidence = log_evidence + float(numpy.log(coverage))
        return self


    @altar.export
    def posterior(self, application: Application) -> typing.Any:
        """
        Sample my posterior distribution, under the {cf_catmip} controller
        """
        from altar.bayesian.controllers.CfCatmip import CfCatmip
        if not isinstance(self.controller, CfCatmip):
            self.error.log("the crossfade model runs under controller = altar.bayesian.cf_catmip")
            raise SystemExit(1)
        return self.controller.posterior(model=self)


    @altar.export
    def initialize_sample(self, step: BayesianState, batch: int | None = None) -> typing.Self:
        """
        Fill {step.theta} with my initial samples, from the conjugate posterior
        """
        θ = self.model.restrict(theta=step.theta)
        rows = θ.shape[0] if batch is None else batch
        # more than were drawn, e.g. a pooled population
        if self.pool.shape[0] < rows:
            self.draw(rows=rows)
        numpy.asarray(θ)[:rows] = self.pool[:rows]
        if self.has_reparametrization:
            step.theta_sampling[...] = step.theta
            self.model.to_sampling(theta=step.theta_sampling, batch=batch)
        return self


    @altar.export
    def likelihoods(self, annealer: Annealer, step: BayesianState,
                    batch: int | None = None) -> typing.Self:
        """
        Fill {step.prior} with log p_conj(θ|d), {step.data} with log p(θ) - log p_conj(θ), and
        {step.posterior} with their combination at {step.beta}
        """
        dispatcher = annealer.dispatcher
        dispatcher.notify(event=dispatcher.prior_start, controller=annealer)
        self.log_densities(theta=self.model.restrict(theta=step.theta),
                           prior=step.prior, data=step.data, batch=batch)
        dispatcher.notify(event=dispatcher.prior_finish, controller=annealer)
        self.eval_posterior(step=step, batch=batch)
        return self


    @altar.export
    def verify(self, step: BayesianState, mask: Array, batch: int | None = None) -> Array:
        """
        The support checks of {model}
        """
        return self.model.verify(step=step, mask=mask, batch=batch)


    @altar.export
    def top(self, annealer: Annealer) -> typing.Self:
        """
        Notification that a β step is about to start, for {model}
        """
        self.model.top(annealer=annealer)
        return self


    @altar.export
    def bottom(self, annealer: Annealer) -> typing.Self:
        """
        Notification that a β step just ended, for {model}
        """
        self.model.bottom(annealer=annealer)
        return self


    @altar.export
    def forward_problem(self, application: Application, theta: typing.Any) -> dict:
        """
        The forward problem of {model}
        """
        return self.model.forward_problem(application=application, theta=theta)


    # the rest of {model}, as the framework reaches it
    def verify_theta(self, theta: Array, mask: Array, batch: int | None = None) -> Array:
        """
        The support checks of {model}, on a bare {theta}
        """
        return self.model.verify_theta(theta=theta, mask=mask, batch=batch)


    def update_model(self, annealer: Annealer, step: BayesianState) -> bool:
        """
        The model uncertainty update of {model}
        """
        return self.model.update_model(annealer=annealer, step=step)


    def eval_prior_with_physical(self, step: BayesianState, likelihood: Array | None = None,
                                 batch: int | None = None) -> typing.Self:
        """
        The log-Jacobian of the reparameterization of {model}
        """
        self.model.eval_prior_with_physical(step=step, likelihood=likelihood, batch=batch)
        return self


    def to_physical(self, theta: Array, batch: int | None = None) -> typing.Self:
        """
        The sampling to physical space transform of {model}
        """
        self.model.to_physical(theta=theta, batch=batch)
        return self


    def to_sampling(self, theta: Array, batch: int | None = None) -> typing.Self:
        """
        The physical to sampling space transform of {model}
        """
        self.model.to_sampling(theta=theta, batch=batch)
        return self


    @property
    def has_reparametrization(self) -> bool:
        """
        Whether {model} is reparameterized
        """
        return self.model.has_reparametrization


    @property
    def psets(self) -> dict:
        """
        The parameter sets of {model}
        """
        return self.model.psets


    @property
    def psets_list(self) -> list:
        """
        The order of the parameter sets of {model}
        """
        return self.model.psets_list


    @property
    def dataobs(self) -> typing.Any:
        """
        The data of {model}
        """
        return self.model.dataobs


    @property
    def cp(self) -> typing.Any:
        """
        The model uncertainty of {model}
        """
        return self.model.cp


    # cross-fade sampling
    def log_densities(self, theta: Array, prior: Array, data: Array,
                      batch: int | None = None) -> typing.Self:
        """
        Fill {prior} with log p_conj(θ|d), and {data} with log p(θ) - log p_conj(θ), where
        log p(θ) is the prior of the parameter sets of {model}
        """
        model = self.model
        rows = theta.shape[0]
        batch = rows if batch is None else batch
        self.conjugate_posterior.log_density(theta=theta, out=prior, batch=batch)
        self._zero(data)
        for name in model.psets_list:
            model.psets[name].eval_prior(theta=theta, prior=data, batch=batch)
        scratch = self._scratch(rows=rows, dtype=data.dtype)
        self.conjugate_prior.log_density(theta=theta, out=scratch, batch=batch)
        if self.cuda:
            altar.cuda.cublas.axpy(alpha=-1.0, x=scratch, y=data, batch=batch)
        else:
            data[:batch] -= scratch[:batch]
        # outside the support of a bounded prior, the target vanishes: no weight, but finite, so
        # that β = 0 still gives the conjugate posterior
        mask = self._zero(self._mask(rows=rows))
        model.verify_theta(theta=theta, mask=mask, batch=batch)
        outside = numpy.asarray(mask)[:batch] != 0
        ratio = numpy.asarray(data)[:batch]
        ratio[outside] = self.floor
        numpy.maximum(ratio, self.floor, out=ratio)
        return self


    def draw(self, rows: int, limit: int = 10**6) -> float:
        """
        Draw {rows} initial samples from the conjugate posterior within the support of the
        prior of {model}, by rejection, the limit of the target as β -> 0; return the fraction
        of the conjugate posterior within the support
        """
        if self.cuda:
            theta = altar.cuda.matrix(shape=(rows, self.parameters), dtype=self.precision)
        else:
            theta = numpy.zeros((rows, self.parameters), dtype=self.precision)
        mask = self._mask(rows=rows)
        kept, count, drawn = [], 0, 0
        while count < rows and drawn < limit:
            candidates = self.conjugate_posterior.sample(rows=rows, rng=self.rng.rng)
            numpy.asarray(theta)[...] = candidates
            self.model.verify_theta(theta=theta, mask=self._zero(mask), batch=rows)
            inside = numpy.asarray(mask)[:rows] == 0
            kept.append(candidates[inside])
            count += int(inside.sum())
            drawn += rows
        if count < rows:
            raise ValueError(
                f"only {count} of {drawn} samples of the conjugate posterior lie within the support "
                f"of the prior; the posterior is too far from the conjugate posterior, try "
                f"a softuniform prior or catmip")
        self.pool = numpy.concatenate(kept)[:rows]
        return count / drawn


    # the log density ratio of the samples outside the support of the prior
    floor = -1e30


    # implementation details
    def _zero(self, array: Array) -> Array:
        """
        Fill {array} with zeroes, on the device or the host
        """
        if self.cuda:
            return array.zero()
        array[...] = 0
        return array


    def _mask(self, rows: int) -> Array:
        """
        A (rows,) mask for the support check of the prior
        """
        if self._flags is None or self._flags.shape[0] != rows:
            if self.cuda:
                self._flags = altar.cuda.vector(shape=rows, dtype="int32").zero()
            else:
                self._flags = numpy.zeros(rows)
        return self._flags


    def _scratch(self, rows: int, dtype: typing.Any) -> Array:
        """
        A (rows,) vector for the conjugate prior
        """
        if self._vector is None or self._vector.shape[0] != rows:
            if self.cuda:
                self._vector = altar.cuda.vector(shape=rows, dtype=dtype).zero()
            else:
                self._vector = numpy.zeros(rows, dtype=dtype)
        return self._vector


    # public data
    conjugate_prior: MultivariateGaussian # N(m, C_m)
    conjugate_posterior: MultivariateGaussian # N(m*, C*)
    log_evidence: float # log p_conj(d) + log q
    pool: numpy.ndarray # the initial samples
    precision: str
    cuda: bool = False

    # private data
    _vector: Array | None = None
    _flags: Array | None = None


# end of file
