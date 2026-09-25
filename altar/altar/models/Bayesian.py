# -*- python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
#
# (c) 2013-2021 parasim inc
# (c) 2010-2021 california institute of technology
# all rights reserved
#


# the package
import altar
# my protocol
from .Model import Model as model


# declaration
class Bayesian(altar.component, family="altar.models.bayesian", implements=model):
    """
    The base class of AlTar models that are compatible with Bayesian explorations
    """


    # user configurable state
    offset = altar.properties.int(default=0)
    offset.doc = "the starting point of my state in the overall controller state"

    parameters = altar.properties.int(default=1)
    parameters.doc = "the number of model degrees of freedom"

    # public data
    rng = None
    controller = None


    # protocol obligations
    @altar.export
    def initialize(self, application):
        """
        Initialize the state of the model given an {application} context
        """
        # get the job parameters
        self.job = application.job
        # borrow the journal channels
        self.info = application.info
        self.warning = application.warning
        self.error = application.error
        self.debug = application.debug
        self.firewall = application.firewall

        # save the random number generator
        self.rng = application.rng
        # and the controller
        self.controller = application.controller

        # all done
        return self


    @altar.export
    def posterior(self, application):
        """
        Sample my posterior distribution
        """
        # ask my controller to help me sample my posterior distribution
        return self.controller.posterior(model=self)


    # services
    @altar.export
    def initialize_sample(self, step, batch=None):
        """
        Fill {step.theta} with an initial random sample from my prior distribution.
        """
        # i don't know what to do, so...


    @altar.export
    def eval_prior(self, step, batch=None):
        """
        Fill {step.prior} with the likelihoods of the samples in {step.theta} in the prior
        distribution
        """
        # i don't know what to do, so...


    @altar.export
    def data_likelihood(self, step, batch=None):
        """
        Fill {step.data} with the likelihoods of the samples in {step.theta} given the available
        data. This is what is usually referred to as the "forward model"
        """
        # i don't know what to do, so...


    @altar.export
    def eval_posterior(self, step, batch=None):
        """
        Given the {step.prior} and {step.data} likelihoods, compute a generalized posterior using
        {step.beta} and deposit the result in {step.post}
        """
        # prime the posterior
        step.posterior.copy(step.prior)
        # compute it; this expression reduces to Bayes' theorem for β->1
        if altar.backends.active() == "cuda":
            # {altar.cuda} is already imported by {altar.backends.activate_cuda}; referencing
            # it here (rather than a fresh `import altar.cuda`) avoids shadowing the
            # module-level {altar} name as a local variable in this function
            altar.cuda.cublas.axpy(alpha=step.beta, x=step.data, y=step.posterior, batch=batch)
        else:
            altar.blas.daxpy(step.beta, step.data, step.posterior)
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
        # compute it
        self.data_likelihood(step=step, batch=batch)
        # done
        dispatcher.notify(event=dispatcher.data_finish, controller=annealer)

        # finally, notify we are about to put together the posterior at this temperature
        dispatcher.notify(event=dispatcher.posterior_start, controller=annealer)
        # compute it
        self.eval_posterior(step=step, batch=batch)
        # done
        dispatcher.notify(event=dispatcher.posterior_finish, controller=annealer)

        # enable chaining
        return self


    @altar.export
    def verify(self, step, mask, batch=None):
        """
        Check whether the samples in {step.theta} are consistent with the model requirements and
        update the {mask}, a vector with zeroes for valid samples and non-zero for invalid ones
        """
        # i don't know what to do, so...
        raise NotImplementedError(
            f"model '{type(self).__name__}' must implement 'verify'")


    def verify_theta(self, theta, mask, batch=None):
        """
        The same check as {verify}, against a bare {theta} matrix instead of a full step --
        for cuda samplers (e.g. Metropolis), which verify a candidate proposal before it has
        been compacted into a full candidate step
        """
        # i don't know what to do, so...
        raise NotImplementedError(
            f"model '{type(self).__name__}' must implement 'verify_theta'")


    # notifications
    @altar.export
    def top(self, annealer):
        """
        Notification that a β step is about to start
        """
        # nothing to do
        return self


    @altar.export
    def bottom(self, annealer):
        """
        Notification that a β step just ended
        """
        # nothing to do
        return self

    @altar.export
    def forward_problem(self, application, theta=None):
        """
        Perform the forward modeling with given {theta}
        """
        # do nothing
        return

    # implementation details
    def mount_input_dataspace(self, pfs):
        """
        Mount the directory with my input files
        """
        # attempt to
        try:
            # mount the directory with my input data
            ifs = altar.filesystem.local(root=self.case)
        # if it fails
        except altar.filesystem.MountPointError as error:
            # grab my error channel
            channel = self.error
            # complain
            channel.log(f"bad case name: '{self.case}'")
            channel.log(str(error))
            # and bail
            raise SystemExit(1)

        # if all goes well, explore it and mount it
        pfs["inputs"] = ifs.discover()
        # all done
        return ifs

    def restrict(self, theta):
        """
        Return my portion of the sample matrix {theta}

        On cpu, {theta} is a real gsl matrix and {.view} is a genuine, offset-preserving
        sub-view -- exactly what {altar.distributions.native.Base.restrict} and
        {altar.models.native.Base.restrict} already rely on for individual parameter sets with
        a non-zero offset.

        On cuda, {theta} is an {altar.cuda.array.Array}: no cuda distribution or parameter set
        downstream of me ever receives a column-sliced sub-view of theta -- each already gets
        the *full* theta grid plus explicit (idx_begin, idx_end) bounds instead (see
        {altar.distributions.cuda.Base.initialize}). The only case that occurs anywhere in
        this codebase today is the identity one: a single, non-embedded model whose {offset}
        is 0 and whose {parameters} span the whole width of {theta}.
        """
        if altar.backends.active() != "cuda":
            # find out how many samples in the set
            samples = theta.rows
            # find where my samples live within the overall sample matrix, and how wide my
            # block is: i own data in all sample rows, starting in the column indicated by my
            # {offset}, with a width determined by my parameter count
            return theta.view(start=(0, self.offset), shape=(samples, self.parameters))

        if self.offset != 0 or self.parameters != theta.shape[1]:
            raise NotImplementedError(
                f"'{type(self).__name__}.restrict': cuda models only support the identity "
                f"case today (offset=0, parameters == theta.shape[1]); got "
                f"offset={self.offset}, parameters={self.parameters}, theta.shape={theta.shape}. "
                f"An embedded/multi-model cuda ensemble would need a real column-sliced Array "
                f"view, which doesn't exist yet.")
        return theta


    # public data
    # job parameters
    job = None
    # journal channels
    info = None
    warning = None
    error = None
    default = None
    firewall = None


# end of file
