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
from .ParameterSet import ParameterSet as parameterset


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

    psets = altar.properties.dict(schema=parameterset())
    psets.default = dict() # empty
    psets.doc = "an ensemble of parameter sets in the model"

    fixed_indices = altar.properties.list(schema=altar.properties.int(), default=None)
    fixed_indices.doc = "the global indices of parameters constrained to fixed values"

    fixed_values = altar.properties.list(schema=altar.properties.float(), default=None)
    fixed_values.doc = "optional per-index fixed values; defaults to {fixed_value}"

    fixed_value = altar.properties.float(default=0.0)
    fixed_value.doc = "default value for constrained parameters when {fixed_values} is not given"

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
    def initializeSample(self, step):
        """
        Fill {step.theta} with an initial random sample from my prior distribution.
        """
        # i don't know what to do, so...


    @altar.export
    def priorLikelihood(self, step):
        """
        Fill {step.prior} with the likelihoods of the samples in {step.theta} in the prior
        distribution
        """
        # i don't know what to do, so...


    @altar.export
    def dataLikelihood(self, step):
        """
        Fill {step.data} with the likelihoods of the samples in {step.theta} given the available
        data. This is what is usually referred to as the "forward model"
        """
        # i don't know what to do, so...


    @altar.export
    def posteriorLikelihood(self, step):
        """
        Given the {step.prior} and {step.data} likelihoods, compute a generalized posterior using
        {step.beta} and deposit the result in {step.post}
        """
        # prime the posterior
        step.posterior.copy(step.prior)
        # compute it; this expression reduces to Bayes' theorem for β->1
        altar.blas.daxpy(step.beta, step.data, step.posterior)
        # all done
        return self


    @altar.export
    def likelihoods(self, annealer, step):
        """
        Convenience function that computes all three likelihoods at once given the current {step}
        of the problem
        """
        # grab the dispatcher
        dispatcher = annealer.dispatcher

        # notify we are about to compute the prior likelihood
        dispatcher.notify(event=dispatcher.priorStart, controller=annealer)
        # compute the prior likelihood
        self.priorLikelihood(step=step)
        # done
        dispatcher.notify(event=dispatcher.priorFinish, controller=annealer)


        # notify we are about to compute the likelihood of the prior given the data
        dispatcher.notify(event=dispatcher.dataStart, controller=annealer)
        # compute it
        self.dataLikelihood(step=step)
        # done
        dispatcher.notify(event=dispatcher.dataFinish, controller=annealer)

        # finally, notify we are about to put together the posterior at this temperature
        dispatcher.notify(event=dispatcher.posteriorStart, controller=annealer)
        # compute it
        self.posteriorLikelihood(step=step)
        # done
        dispatcher.notify(event=dispatcher.posteriorFinish, controller=annealer)

        # enable chaining
        return self


    @altar.export
    def verify(self, step, mask):
        """
        Check whether the samples in {step.theta} are consistent with the model requirements and
        update the {mask}, a vector with zeroes for valid samples and non-zero for invalid ones
        """
        # i don't know what to do, so...
        raise NotImplementedError(
            f"model '{type(self).__name__}' must implement 'verify'")


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
    def forwardProblem(self, application, theta=None):
        """
        Perform the forward modeling with given {theta}
        """
        # do nothing
        return

    @altar.export
    def configureFixedParameters(self, parameters=None):
        """
        Validate and cache the specification of fixed parameters.
        """
        # resolve the parameter count
        parameters = self.parameters if parameters is None else parameters
        # normalize the index list
        indices = self.fixed_indices
        indices = [] if indices is None else list(indices)
        # no fixed parameters
        if len(indices) == 0:
            self._fixedParameterMap = tuple()
            self._fixedParameterCount = parameters
            return self

        # choose a value for each index
        values = self.fixed_values
        if values is None:
            values = [self.fixed_value] * len(indices)
        else:
            values = list(values)
            if len(values) == 1 and len(indices) > 1:
                values = values * len(indices)
            elif len(values) != len(indices):
                raise ValueError(
                    f"model '{type(self).__name__}': expected either one fixed value or "
                    f"{len(indices)} values for {len(indices)} fixed indices; got {len(values)}")

        # validate and deduplicate
        mapping = {}
        for index, value in zip(indices, values):
            i = int(index)
            if i < 0 or i >= parameters:
                raise ValueError(
                    f"model '{type(self).__name__}': fixed index {i} is out of range "
                    f"for {parameters} parameters")
            v = float(value)
            if i in mapping and mapping[i] != v:
                raise ValueError(
                    f"model '{type(self).__name__}': conflicting fixed values for index {i}")
            mapping[i] = v

        # store a deterministic ordered representation
        self._fixedParameterMap = tuple(sorted(mapping.items()))
        self._fixedParameterCount = parameters
        return self

    @altar.export
    def deduceFixedParameters(self, psets=None, parameters=None):
        """
        Merge explicit fixed-parameter settings with fixed parameters implied by priors.
        """
        # resolve my geometry
        parameters = self.parameters if parameters is None else parameters
        psets = self.psets if psets is None else psets

        # start from the explicitly configured constraints
        self.configureFixedParameters(parameters=parameters)
        mapping = dict(self._fixedParameterMap)

        # collect fixed parameters from each prior that can provide them
        for pset in psets.values():
            prior = getattr(pset, "prior", None)
            if prior is None or not hasattr(prior, "fixedParameterMap"):
                continue

            local = prior.fixedParameterMap(offset=pset.offset, parameters=pset.count)
            for index, value in local:
                i = int(index)
                v = float(value)
                if i < 0 or i >= parameters:
                    raise ValueError(
                        f"model '{type(self).__name__}': prior fixed index {i} is out of range "
                        f"for {parameters} parameters")
                if i in mapping and mapping[i] != v:
                    raise ValueError(
                        f"model '{type(self).__name__}': conflicting fixed values for index {i}")
                mapping[i] = v

        # store merged constraints
        self._fixedParameterMap = tuple(sorted(mapping.items()))
        self._fixedParameterCount = parameters
        if len(self._fixedParameterMap) > 0:
            self.fixed_indices = [index for index, _ in self._fixedParameterMap]
            self.fixed_values = [value for _, value in self._fixedParameterMap]

        # all done
        return self

    @altar.export
    def fixedParameterMap(self, parameters=None):
        """
        Return the fixed parameter map as a tuple of ``(index, value)`` pairs.
        """
        parameters = self.parameters if parameters is None else parameters
        if self._fixedParameterMap is None or self._fixedParameterCount != parameters:
            self.configureFixedParameters(parameters=parameters)
        return self._fixedParameterMap

    @altar.export
    def fixedParameterIndices(self, parameters=None):
        """
        Return the tuple of constrained parameter indices.
        """
        return tuple(index for index, _ in self.fixedParameterMap(parameters=parameters))

    @altar.export
    def applyFixedParameters(self, theta):
        """
        Project fixed parameters in {theta} onto their prescribed values.
        """
        # this projection is implemented for host matrices
        if not hasattr(theta, "rows") or not hasattr(theta, "columns"):
            return theta

        # resolve geometry
        samples = theta.rows
        parameters = theta.columns
        fixed = self.fixedParameterMap(parameters=parameters)

        # quick exit
        if len(fixed) == 0:
            return theta

        # overwrite constrained entries in all samples
        for sample in range(samples):
            for index, value in fixed:
                theta[sample, index] = value

        # all done
        return theta

    # implementation details
    def mountInputDataspace(self, pfs):
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
        """
        # find out how many samples in the set
        samples = theta.rows
        # get my parameter count
        parameters = self.parameters
        # get my offset in the samples
        offset = self.offset

        # find where my samples live within the overall sample matrix:
        start = 0, offset
        # form the shape of the sample matrix that's mine
        shape = samples, parameters

        # return a view to the portion of the sample that's mine: i own data in all sample
        # rows, starting in the column indicated by my {offset}, and the width of my block is
        # determined by my parameter count
        return theta.view(start=start, shape=shape)


    # public data
    # job parameters
    job = None
    # journal channels
    info = None
    warning = None
    error = None
    default = None
    firewall = None

    _fixedParameterMap = None
    _fixedParameterCount = None


# end of file
