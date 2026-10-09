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
# my protocol
from .Bayesian import Bayesian
# the model uncertainty
from .cp import cp as uncertainty

if typing.TYPE_CHECKING:
    from altar.arrays import Array
    from altar.bayesian.controllers.Annealer import Annealer
    from altar.bayesian.states.BayesianState import BayesianState
    from altar.shells.Application import Application

# declaration
class BayesianL2(Bayesian, family="altar.models.bayesianl2"):
    """
    A (Simplified) Bayesian Model with ParameterSets and L2 data norm
    """

    # user configurable state

    parameters = altar.properties.int(default=1)
    parameters.doc = "the number of model degrees of freedom"

    cascaded = altar.properties.bool(default=False)
    cascaded.doc = "whether the model is cascaded (annealing temperature is fixed at 1)"

    embedded = altar.properties.bool(default=False)
    embedded.doc = "whether the model is embedded in an ensemble of models"

    psets_list = altar.properties.list(schema=altar.properties.str(), default=None)
    psets_list.doc = "the order in which {psets} are laid out in the overall parameter " \
                 "vector; required when {psets} is non-empty, since {psets} itself is a " \
                 "dict and doesn't guarantee iteration order"

    psets = altar.properties.dict(schema=altar.models.parameters())
    psets.default = dict() # empty
    psets.doc = "an ensemble of parameter sets in the model"

    dataobs = altar.data.data()
    dataobs.default = altar.data.datal2()
    dataobs.doc = "observed data"

    cp = uncertainty()
    cp.doc = "the model uncertainty C_p added to the data covariance: none, fixed or adaptive"

    # the path of input files
    case = altar.properties.path(default="input")
    case.doc = "the directory with the input files"

    idx_map=altar.properties.list(schema=altar.properties.int())
    idx_map.default = None
    idx_map.doc = "the indices for model parameters in whole theta set"

    return_residual = altar.properties.bool(default=True)
    return_residual.doc = "the forward model returns residual(True) or prediction(False)"

    # protocol obligations
    @altar.export
    def initialize(self, application: Application) -> typing.Self:
        """
        Initialize the state of the model given an {application} context
        """
        # super class method
        super().initialize(application=application)

        # the precision of my backend, for my inputs, my predictions and my own numerics
        self.precision = application.job.working_precision

        # mount my input data space
        self.ifs = self.mount_input_dataspace(pfs=application.pfs)
        # set up my file reader/writer
        self.io = altar.io.FileIO(ifs=self.ifs, error=self.error, precision=self.precision)
        # find out how many samples I will be working with; this equal to the number of chains
        self.samples = application.job.chains

        # initialize the data
        self.dataobs.initialize(application=application)
        self.observations = self.dataobs.observations

        # lay out my parameter sets, in {psets_list} order, and let each one initialize
        # itself; the total number of parameters is now known, so record it; in an ensemble,
        # the ensemble owns the parameter sets and has already set my {parameters}
        if not self.embedded:
            self.parameters = self.initialize_psets(application=application)

        # the model uncertainty, e.g. a fixed C_p folded into the data covariance right away
        self.cp.initialize(model=self, application=application)

        # all done
        return self

    @altar.export
    def posterior(self, application: Application) -> typing.Any:
        """
        Sample my posterior distribution
        """
        # ask my controller to help me sample my posterior distribution
        return self.controller.posterior(model=self)

    @altar.export
    def initialize_sample(self, step: BayesianState, batch: int | None = None) -> typing.Self:
        """
        Fill {step.θ} with an initial random sample from my prior distribution, or, in
        cross-fade sampling, from my conjugate posterior
        """
        # grab the portion of the sample that's mine
        θ = self.restrict(theta=step.theta)
        if self._crossfade is not None:
            rows = θ.shape[0] if batch is None else batch
            # more than were drawn, e.g. a pooled population
            if self._crossfade.pool.shape[0] < rows:
                self._crossfade.draw(model=self, rows=rows)
            numpy.asarray(θ)[:rows] = self._crossfade.pool[:rows]
            if self.has_reparametrization:
                step.theta_sampling[...] = step.theta
                self.to_sampling(theta=step.theta_sampling, batch=batch)
            return self
        # go through each parameter set, in {psets_list} order -- {psets} is a dict and may
        # carry extra entries merged in from other configuration sources
        for name in self.psets_list:
            pset = self.psets[name]
            # and ask each one to {prep} the sample; always in physical space -- {step.theta}
            # is the one buffer every sampler (Metropolis, SGLD, HMC) agrees means physical
            pset.initialize_sample(theta=θ, batch=batch)
        # bridge into sampling space, once, for a reparameterized model: seed
        # {step.theta_sampling} as a copy of the physical draw, then transform in place --
        # {to_sampling} only actually does anything for the psets that are reparameterized,
        # leaving the rest as the (correct, identity) copy
        if self.has_reparametrization:
            step.theta_sampling[...] = step.theta
            self.to_sampling(theta=step.theta_sampling, batch=batch)
        # and return
        return self

    @altar.export
    def verify(self, step: BayesianState, mask: Array, batch: int | None = None) -> Array:
        """
        Check whether the samples in {step.theta} are consistent with the model requirements and
        update the {mask}, a vector with zeroes for valid samples and non-zero for invalid ones
        """
        return self.verify_theta(theta=step.theta, mask=mask, batch=batch)


    def verify_theta(self, theta: Array, mask: Array, batch: int | None = None) -> Array:
        """
        The same check as {verify}, against a bare {theta} matrix instead of a full step --
        for cuda samplers (e.g. Metropolis), which verify a candidate proposal before it has
        been compacted into a full candidate step
        """
        # grab the portion of the sample that's mine
        θ = self.restrict(theta=theta)
        # ask my subsets, in {psets_list} order
        for name in self.psets_list:
            pset = self.psets[name]
            # and ask each one to verify the sample
            pset.verify(theta=θ, mask=mask, batch=batch)
        # all done; return the rejection map
        return mask

    @altar.export
    def eval_prior(self, step: BayesianState, batch: int | None = None) -> typing.Self:
        """
        Fill {step.prior} with the log likelihoods of the samples in {step.theta} in my prior
        distribution
        """
        # grab the portion of the sample that's mine
        θ = self.restrict(theta=step.theta)
        # ask my subsets, in {psets_list} order
        for name in self.psets_list:
            pset = self.psets[name]
            # and ask each one to evaluate the prior
            pset.eval_prior(theta=θ, prior=step.prior, batch=batch)

        # all done
        return self


    def eval_prior_with_physical(self, step: BayesianState, likelihood: Array | None = None,
                                 batch: int | None = None) -> typing.Self:
        """
        Add the log-Jacobian of every reparameterized pset into {likelihood} (default
        {step.prior}); samplers keep it in a separate per-sample buffer so {step.prior} stays
        the physical-space prior
        """
        likelihood = step.prior if likelihood is None else likelihood
        # grab the portion of the sample that's mine
        θ = self.restrict(theta=step.theta)
        # ask my subsets, in {psets_list} order
        for name in self.psets_list:
            pset = self.psets[name]
            pset.eval_prior_with_physical(theta=θ, prior=likelihood, batch=batch)
        # all done
        return self


    def eval_prior_physical(self, step: BayesianState, batch: int | None = None) -> typing.Self:
        """
        Fill {step.prior} with the log likelihoods of the samples in {step.theta}, given in
        physical space
        """
        # grab the portion of the sample that's mine
        θ = self.restrict(theta=step.theta)
        # ask my subsets, in {psets_list} order
        for name in self.psets_list:
            pset = self.psets[name]
            pset.eval_prior_physical(theta=θ, prior=step.prior, batch=batch)
        # all done
        return self


    def to_physical(self, theta: Array, batch: int | None = None) -> typing.Self:
        """
        Transform {theta} from sampling space to physical space, in place; only
        reparameterized psets actually do anything here
        """
        # grab the portion of the sample that's mine
        θ = self.restrict(theta=theta)
        # ask my subsets, in {psets_list} order
        for name in self.psets_list:
            self.psets[name].to_physical(theta=θ, batch=batch)
        # all done
        return self


    def to_sampling(self, theta: Array, batch: int | None = None) -> typing.Self:
        """
        Transform {theta} from physical space to sampling space, in place; the inverse of
        {to_physical}
        """
        # grab the portion of the sample that's mine
        θ = self.restrict(theta=theta)
        # ask my subsets, in {psets_list} order
        for name in self.psets_list:
            self.psets[name].to_sampling(theta=θ, batch=batch)
        # all done
        return self


    def eval_jacobian(self, step: BayesianState, batch: int | None = None) -> typing.Self:
        """
        Fill {step.Jacobian} with d(physical)/d(sampling) for every reparameterized pset (1
        elsewhere), for use by a reparameterized gradient-based sampler (e.g. HMC)
        """
        # grab the portion of the sample that's mine
        θ = self.restrict(theta=step.theta)
        # ask my subsets, in {psets_list} order
        for name in self.psets_list:
            self.psets[name].jacobian(theta=θ, jacobian=step.Jacobian, batch=batch)
        # all done
        return self


    def transformToPhysical(self, step: BayesianState) -> typing.Self:
        """
        Refresh {step.theta} (physical space) from {step.phi} (sampling space); the bridge a
        reparameterized gradient-based sampler (e.g. cuda {HMC}) needs after every leapfrog
        position update
        """
        step.theta[...] = step.phi
        self.to_physical(theta=step.theta, batch=step.samples)
        return self


    @property
    def has_reparametrization(self) -> bool:
        """
        Whether any of my psets' priors are reparameterized; read by
        {altar.bayesian.states.cuda.CoolingStep.start}/my own {initialize_sample} to decide
        whether to carry the extra sampling/physical-space state.

        Reads each prior's {reparameterize} *configuration* trait, not its
        {has_reparametrization} attribute -- the latter is only set once the prior's own
        {initialize} has run (inside {self.initialize}/{initialize_psets}), but
        {Application.main} calls {controller.initialize} (and therefore this, transitively,
        via {CoolingStep.allocate}/{HMCState.start}, which read this property through the
        model instance before it is initialized) *before* {model.initialize} -- so this must
        stay a plain config read, not a mirror of state set up later.
        """
        return any(getattr(self.psets[name].prior, 'reparameterize', False)
                   for name in self.psets_list)


    @property
    def reparameterization(self) -> bool:
        """
        Alias of {has_reparametrization}; this is the name cuda {HMC.initialize} reads
        """
        return self.has_reparametrization


    def forward_model(self, theta: numpy.ndarray, prediction: numpy.ndarray) -> typing.Self:
        """
        The forward model for a single set of parameters: fill {prediction} from {theta}
        """
        # i don't know what to do, so...
        raise NotImplementedError(
            f"model '{type(self).__name__}' must implement 'forward_model'")


    def forward_model_batched(self, theta: Array, prediction: Array,
                              batch: int | None = None) -> typing.Self:
        """
        Fill the first {batch} rows of the (samples x observations) {prediction} with the
        predictions, or the residuals if {return_residual}, of the samples in {theta}
        """
        # the default asks {forward_model}, one sample at a time, on the cpu only
        if altar.backends.active() == "cuda":
            raise NotImplementedError(
                f"model '{type(self).__name__}' must implement 'forward_model_batched' on cuda")
        batch = theta.shape[0] if batch is None else batch
        for sample in range(batch):
            self.forward_model(theta=theta[sample], prediction=prediction[sample])
        # all done
        return self


    def eval_data_likelihood(self, theta: Array, likelihood: Array,
                             batch: int | None = None) -> typing.Self:
        """
        calculate data likelihood and add it to step.prior or step.data
        """
        # This method assumes that there is a forward_model_batched defined
        # Otherwise, please define your own version of this method

        # a matrix for the prediction (samples, observations), made once and reused
        prediction = self._prediction
        if prediction is None:
            if altar.backends.active() == "cuda":
                # {altar.cuda} is already imported by {altar.backends.activate_cuda}; referencing
                # it here (rather than a fresh `import altar.cuda`) avoids shadowing the
                # module-level {altar} name as a local variable in this function
                prediction = altar.cuda.matrix(shape=(self.samples, self.observations), dtype=self.precision)
            else:
                prediction = numpy.zeros((self.samples, self.observations), dtype=self.precision)
            self._prediction = prediction
        # survey forward model whether it computes residual or not
        returnResidual = self.return_residual
        # call forward_model to calculate the data prediction or its difference between dataobs
        self.forward_model_batched(theta=theta, prediction=prediction, batch=batch)
        # call data to calculate the l2 norm; a raw prediction never has the covariance merged in
        self.dataobs.eval_likelihood(
            prediction=prediction, likelihood=likelihood, residual=returnResidual, batch=batch,
            whitened=returnResidual)

        # all done
        return self


    @altar.export
    def likelihoods(self, annealer: Annealer, step: BayesianState,
                    batch: int | None = None) -> typing.Self:
        """
        Convenience function that computes all three likelihoods at once given the current {step}
        of the problem
        """

        # grab the dispatcher
        dispatcher = annealer.dispatcher

        # cross-fade sampling: the conjugate posterior, and the ratio of my prior to the
        # conjugate prior, in place of my prior and my data likelihood
        if self._crossfade is not None:
            dispatcher.notify(event=dispatcher.prior_start, controller=annealer)
            self._crossfade.log_densities(model=self, theta=self.restrict(theta=step.theta),
                                          prior=step.prior, data=step.data, batch=batch)
            dispatcher.notify(event=dispatcher.prior_finish, controller=annealer)
            self.eval_posterior(step=step, batch=batch)
            return self

        # notify we are about to compute the prior likelihood
        dispatcher.notify(event=dispatcher.prior_start, controller=annealer)
        # compute the prior likelihood
        self.eval_prior(step=step, batch=batch)
        # done
        dispatcher.notify(event=dispatcher.prior_finish, controller=annealer)

        # notify we are about to compute the likelihood of the prior given the data
        dispatcher.notify(event=dispatcher.data_start, controller=annealer)

        # grab the portion of the sample that's mine
        θ = self.restrict(theta=step.theta)
        # compute it
        self.eval_data_likelihood(theta=θ, likelihood=step.data, batch=batch)
        # done
        dispatcher.notify(event=dispatcher.data_finish, controller=annealer)

        # finally, notify we are about to put together the posterior at this temperature
        dispatcher.notify(event=dispatcher.posterior_start, controller=annealer)
        # compute it; inherited from {Bayesian}, already backend-dispatching
        self.eval_posterior(step=step, batch=batch)
        # done
        dispatcher.notify(event=dispatcher.posterior_finish, controller=annealer)

        # enable chaining
        return self


    def update_model(self, annealer: Annealer, step: BayesianState) -> bool:
        """
        At the start of a walk at a new beta, let my model uncertainty update C_chi; return True
        if it changed, so the densities of {step} get recomputed
        """
        return self.cp.update(model=self, annealer=annealer, step=step)


    def update_covariance(self, cp: numpy.ndarray | None = None) -> typing.Self:
        """
        Set the data likelihood's covariance to C_chi = C_d + {cp}, a numpy (observations x
        observations) array, or back to C_d alone
        """
        self.dataobs.update_covariance(cp=cp)
        self.covariance_updated()
        return self


    def covariance_updated(self) -> typing.Self:
        """
        Notification that the data covariance changed, for models that fold it into their own
        data, e.g. premerged green's functions
        """
        return self


    def compute_cp(self, theta: numpy.ndarray) -> numpy.ndarray:
        """
        The model uncertainty C_p, (observations x observations), for the mean model {theta};
        models that can estimate it override this
        """
        raise NotImplementedError(
            f"model '{type(self).__name__}' cannot estimate C_p from a mean model; "
            f"use a fixed C_p (cp=altar.models.cp.fixed) instead")


    # implementation details
    # {mount_input_dataspace}/{restrict} are inherited unchanged from {Bayesian}

    def initialize_psets(self, application: Application | None = None) -> int:
        """
        Lay out my {psets} one after another, in {psets_list} order -- {psets} is a dict and
        doesn't guarantee iteration order matches the user's declared layout -- and let each
        one initialize itself. Returns the total number of parameters they cover.
        """
        # accumulate the offset as we go
        offset = 0
        # in the user-declared order
        for name in self.psets_list:
            # get the parameter set
            pset = self.psets[name]
            # and let it initialize itself at the current offset
            offset += pset.initialize(model=self, offset=offset, application=application)
        # all done
        return offset

    def verify_unbounded_priors(self) -> typing.Self:
        """
        Raise if any active prior is bounded and not reparameterized: gradient-based samplers
        (HMC, MALA, SGLD) move the chains along the gradient of the posterior, which a bounded
        prior (e.g. a uniform one) doesn't define on the whole real line, so it needs an
        unconstrained reparameterization (see {altar.distributions.Uniform.reparameterize})
        before it can be sampled this way
        """
        bounded = [self.psets[name].prior for name in self.psets_list
                   if self.psets[name].prior.bounded
                   and not getattr(self.psets[name].prior, 'reparameterize', False)]
        if bounded:
            channel = self.error
            names = ", ".join(type(p).__name__ for p in bounded)
            channel.log(
                f"gradient-based samplers (HMC, MALA, SGLD) only support unbounded priors; "
                f"found bounded prior(s): {names}. Use CATMIP/Metropolis for this model, "
                f"or set reparameterize=True on the prior.")
            raise SystemExit(1)
        self.checked_unbounded_priors = True
        return self

    # cross-fade sampling (Minson, 2024)
    def crossfade(self) -> float:
        """
        Switch to cross-fade sampling: draw the initial samples from my conjugate posterior,
        within the support of my prior, and evaluate the conjugate posterior and the ratio of my
        prior to the conjugate prior in place of my prior and my data likelihood; return the
        evidence of the conjugate model within the support, log p_conj(d) + log q, with q the
        fraction of the conjugate posterior within the support
        """
        from .CrossFade import CrossFade
        # the multivariate normal distribution of my backend
        backend = "cuda" if altar.backends.active() == "cuda" else "native"
        MultivariateGaussian = import_module(
            f"altar.distributions.{backend}.MultivariateGaussian").MultivariateGaussian
        if self.embedded:
            raise NotImplementedError("cross-fade sampling of a model in an ensemble")
        mean, variance = self.conjugate_prior()
        mstar, cstar, log_evidence = self.conjugate_posterior(mean=mean, variance=variance)
        precision = self.precision
        self._crossfade = CrossFade(
            prior=MultivariateGaussian(
                mean=mean, covariance=numpy.diag(variance), precision=precision),
            posterior=MultivariateGaussian(mean=mstar, covariance=cstar, precision=precision),
            log_evidence=log_evidence, rng=self.rng.rng, precision=precision)
        coverage = self._crossfade.draw(model=self, rows=self.samples)
        return log_evidence + float(numpy.log(coverage))


    def conjugate_prior(self) -> tuple[numpy.ndarray, numpy.ndarray]:
        """
        The mean and the variance of each of my parameters under the conjugate prior, a normal
        distribution matched to the mean and the variance of the prior of its parameter set
        """
        mean = numpy.zeros(self.parameters)
        variance = numpy.zeros(self.parameters)
        for name in self.psets_list:
            pset = self.psets[name]
            prior = pset.prior
            columns = slice(pset.offset, pset.offset + pset.count)
            if hasattr(prior, "mean") and hasattr(prior, "sigma"):
                mean[columns], variance[columns] = prior.mean, prior.sigma ** 2
            elif hasattr(prior, "support"):
                low, high = prior.support
                mean[columns], variance[columns] = (low + high) / 2, (high - low) ** 2 / 12
            else:
                raise NotImplementedError(
                    f"no conjugate prior for the {type(prior).__name__} prior of '{name}'")
        return mean, variance


    def conjugate_posterior(self, mean: numpy.ndarray,
                            variance: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray, float]:
        """
        The mean m* and the covariance C* of my posterior under the conjugate prior N(mean,
        diag(variance)), and the evidence of that model, log p_conj(d); models that support
        cross-fade sampling provide them
        """
        raise NotImplementedError(
            f"'{type(self).__name__}' has no conjugate posterior, for cross-fade sampling")


    # private data
    _crossfade: typing.Any = None # my cross-fade state, when sampled by cross-fading
    observations: int
    device: typing.Any = None
    precision: str
    ifs: altar.filesystem.Filesystem.Filesystem # the filesystem with the input files
    io: altar.io.FileIO # my file reader/writer
    samples: int
    checked_unbounded_priors: bool = False # whether {gradient} has already verified all priors are unbounded
    _prediction: Array | None = None # the scratch prediction of {eval_data_likelihood}


# end of file
