# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-2021 parasim inc
# (c) 2010-2021 california institute of technology
# all rights reserved
#
# Author(s): Lijun Zhu


# the package
import altar
# my protocol
from .Bayesian import Bayesian

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
    def initialize(self, application):
        """
        Initialize the state of the model given an {application} context
        """
        # super class method
        super().initialize(application=application)

        # my working precision, needed by {eval_data_likelihood}'s cuda buffer allocation and
        # by any concrete model's own cuda numerics (mirrors
        # {altar.distributions.cuda.Base.initialize})
        self.precision = application.job.gpuprecision

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
        # itself; the total number of parameters is now known, so record it
        self.parameters = self.initialize_psets(application=application)

        # all done
        return self

    @altar.export
    def posterior(self, application):
        """
        Sample my posterior distribution
        """
        # ask my controller to help me sample my posterior distribution
        return self.controller.posterior(model=self)

    @altar.export
    def initialize_sample(self, step, batch=None):
        """
        Fill {step.θ} with an initial random sample from my prior distribution.
        """
        # grab the portion of the sample that's mine
        θ = self.restrict(theta=step.theta)
        # go through each parameter set, in {psets_list} order -- {psets} is a dict and may
        # carry extra entries merged in from other configuration sources
        for name in self.psets_list:
            pset = self.psets[name]
            # and ask each one to {prep} the sample
            pset.initialize_sample(theta=θ, batch=batch)
        # and return
        return self

    @altar.export
    def verify(self, step, mask, batch=None):
        """
        Check whether the samples in {step.theta} are consistent with the model requirements and
        update the {mask}, a vector with zeroes for valid samples and non-zero for invalid ones
        """
        return self.verify_theta(theta=step.theta, mask=mask, batch=batch)


    def verify_theta(self, theta, mask, batch=None):
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
    def eval_prior(self, step, batch=None):
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


    def forward_model(self, theta, prediction):
        """
        The forward model for a single set of parameters
        """
        # i don't know what to do, so...
        raise NotImplementedError(
            f"model '{type(self).__name__}' must implement 'forward_model'")


    def forward_model_batched(self, theta, prediction):
        """
        The forward model for a batch of theta: compute prediction from theta
        also return {residual}=True, False if the difference between data and prediction is computed
        """

        # The default method computes samples one by one
        batch = self.samples
        # create a prediction vector
        prediction_sample = altar.vector(shape=self.observations)
        # iterate over samples
        for sample in range(batch):
            # obtain the sample (one set of parameters)
            theta_sample = theta.getRow(sample)
            # call the forward model
            self.forward_model(theta=theta_sample, prediction=prediction_sample)
            # copy to the prediction matrix
            prediction.setRow(sample, prediction_sample)

        # all done
        return self


    def eval_data_likelihood(self, theta, likelihood, batch=None):
        """
        calculate data likelihood and add it to step.prior or step.data
        """
        # This method assumes that there is a forward_model_batched defined
        # Otherwise, please define your own version of this method

        # create a matrix for the prediction (samples, observations)
        if altar.backends.active() == "cuda":
            # {altar.cuda} is already imported by {altar.backends.activate_cuda}; referencing
            # it here (rather than a fresh `import altar.cuda`) avoids shadowing the
            # module-level {altar} name as a local variable in this function
            prediction = altar.cuda.matrix(shape=(self.samples, self.observations), dtype=self.precision)
        else:
            prediction = altar.matrix(shape=(self.samples, self.observations))
        # survey forward model whether it computes residual or not
        returnResidual = self.return_residual
        # call forward_model to calculate the data prediction or its difference between dataobs
        self.forward_model_batched(theta=theta, prediction=prediction, batch=batch)
        # call data to calculate the l2 norm
        self.dataobs.eval_likelihood(
            prediction=prediction, likelihood=likelihood, residual=returnResidual, batch=batch)

        # all done
        return self


    @altar.export
    def likelihoods(self, annealer, step, batch=None):
        """
        Convenience function that computes all three likelihoods at once given the current {step}
        of the problem
        """

        # grab the dispatcher
        dispatcher = annealer.dispatcher

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


    def update_model(self, annealer):
        """
        Update Model parameters if needed
        :param annealer:
        :return: default is False
        """
        return False


    # implementation details
    # {mount_input_dataspace}/{restrict} are inherited unchanged from {Bayesian}

    def initialize_psets(self, application=None):
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

    def verify_unbounded_priors(self):
        """
        Raise if any active prior is bounded: gradient-based samplers (SGLD, HMC) have no
        accept/reject step, so a bounded prior (e.g. a uniform one) needs an unconstrained
        reparameterization -- not yet implemented -- before it can be sampled this way
        """
        bounded = [self.psets[name].prior for name in self.psets_list
                   if self.psets[name].prior.bounded]
        if bounded:
            channel = self.error
            names = ", ".join(type(p).__name__ for p in bounded)
            channel.log(
                f"gradient-based samplers (SGLD, HMC) only support unbounded priors; "
                f"found bounded prior(s): {names}. Use CATMIP/Metropolis for this model, "
                f"or wait for logistic-transform support.")
            raise SystemExit(1)
        self.checked_unbounded_priors = True
        return self

    # private data
    observations = None
    device = None
    precision = None
    ifs = None # the filesystem with the input files
    io = None # my file reader/writer
    checked_unbounded_priors = False # whether {gradient} has already verified all priors are unbounded


# end of file
