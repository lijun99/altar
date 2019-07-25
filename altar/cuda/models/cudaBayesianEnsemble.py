# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-2019 parasim inc
# (c) 2010-2019 california institute of technology
# all rights reserved
#
# Author(s): Lijun Zhu


# the package
import altar
import altar.cuda

# my superclass
from altar.models.Bayesian import Bayesian


# declaration
class cudaBayesianEnsemble(Bayesian, family="altar.models.cudaensemble"):
    """
    A collection of AlTar models that comprise a single model
    """

    # my collection
    models = altar.properties.dict(schema=altar.cuda.models.model())
    models.doc = "the collection of models in this ensemble"

    parameters = altar.properties.int(default=1)
    parameters.doc = "the number of model degrees of freedom"

    psets_list = altar.properties.list(default=None)
    psets_list.doc = "list of parameter sets, used to set orders"

    psets = altar.properties.dict(schema=altar.cuda.models.parameters())
    psets.doc = "an ensemble of parameter sets in the model"

    # the path of input files
    case = altar.properties.path(default="input")
    case.doc = "the directory with the input files"

    # protocol obligations
    @altar.export
    def initialize(self, application):
        """
        Initialize the state of the model given an {application} context
        """
        # chain up
        super().initialize(application=application)

        # mount my input data space
        self.ifs = self.mountInputDataspace(pfs=application.pfs)

        # find out how many samples I will be working with; this equal to the number of chains
        self.samples = application.job.chains

        # cuda method
        self.device = application.controller.worker.device
        self.precision = application.job.gpuprecision

        # initialize the parametersets
        # initialize the offset
        parameters = 0
        # go through my parameter sets
        for name in self.psets_list:
            # get the parameter set from psets dictionary
            pset = self.psets[name]
            # set the offset
            pset.offset = parameters
            # initialize the pset
            parameters += pset.cuInitialize(application=application)
        self.parameters = parameters

        # go through my models
        for name, model in self.models.items():
            # and initialize each one
            model.embedded = True
            model.initialize(application=application)

        self.cuInitialize(application=application)

        self.datallk = altar.cuda.vector(shape=self.samples, dtype=self.precision)
        # all done
        return self

    def cuInitialize(self, application):
        """
        cuda initialization
        """
        return self

    @altar.export
    def posterior(self, application):
        """
        Sample my posterior distribution
        """
        # ask my controller to help me sample my posterior distribution
        return self.controller.posterior(model=self)

    def cuInitSample(self, theta):
        """
        Fill {theta} with an initial random sample from my prior distribution.
        """
        # ask my subsets
        for name, pset in self.psets.items():
            # and ask each one to verify the sample
            pset.prep.cuInitSample(theta=theta)

        # all done
        return self

    def cuVerify(self, theta, mask):
        """
        Check whether the samples in {step.theta} are consistent with the model requirements and
        update the {mask}, a vector with zeroes for valid samples and non-zero for invalid ones
        """
        # ask my subsets
        for pset in self.psets.values():
            # and ask each one to verify the sample
            pset.prior.cuVerify(theta=theta, mask=mask)
        # all done; return the rejection map
        return mask

    def cuEvalPrior(self, theta, prior, batch):
        """
        Fill {priorLLK} with the log likelihoods of the samples in {theta} in my prior distribution
        """
        batch = batch if batch is not None else theta.rows
        # ask my subsets
        for pset in self.psets.values():
            # and ask each one to verify the sample
            pset.prior.cuEvalPrior(theta=theta, prior=prior, batch=batch)

        # all done
        return self

    def cuEvalLikelihood(self, step, batch):
        """
        Fill {step.data} with the likelihoods of the samples in {step.theta} given the available
        data. This is what is usually referred to as the "forward model"
        """
        datallk = self.datallk
        # ask each of my models
        for name, model in self.models.items():
            # to contribute to the computation of the data likelihood

            # make a local copy of theta if needed
            # mtheta = model.restricted(theta=step.theta, batch=batch)

            model.cuEvalLikelihood(theta=step.theta, likelihood=datallk.zero(), batch=batch)
            if model.cascaded:
                step.prior += datallk
            else:
                step.data += datallk
            #datallk.print()
            #step.prior.print()
            #step.data.print()

        # all done
        return self

    def cuEvalPosterior(self, step, batch):
        """
        Given the {step.prior} and {step.data} likelihoods, compute a generalized posterior using
        {step.beta} and deposit the result in {step.post}
        """
        # prime the posterior
        step.posterior.copy(step.prior)
        # compute it; this expression reduces to Bayes' theorem for β->1
        altar.cuda.cublas.axpy(alpha=step.beta, x=step.data, y=step.posterior, batch=batch)
        # all done
        return self

    def updateModel(self, annealer):
        """
        Update model parameters if needed
        :param annealer:
        :return:
        """
        # default is not updated
        out = False
        # iterate over embedded models
        for name, model in self.models.item():
            updated = model.updateModel(annealer=annealer)
            out = out or updated
        # all done
        return out

    @altar.export
    def likelihoods(self, annealer, step, batch):
        """
        Convenience function that computes all three likelihoods at once given the current {step}
        of the problem
        """
        batch = step.samples if batch is None else batch

        # grab the dispatcher
        dispatcher = annealer.dispatcher

        # notify we are about to compute the prior likelihood
        dispatcher.notify(event=dispatcher.priorStart, controller=annealer)
        # compute the prior likelihood
        self.cuEvalPrior(theta=step.theta, prior=step.prior, batch=batch)
        # done
        dispatcher.notify(event=dispatcher.priorFinish, controller=annealer)

        # notify we are about to compute the likelihood of the prior given the data
        dispatcher.notify(event=dispatcher.dataStart, controller=annealer)
        # compute it
        self.cuEvalLikelihood(step=step, batch=batch)
        # done
        dispatcher.notify(event=dispatcher.dataFinish, controller=annealer)

        # finally, notify we are about to put together the posterior at this temperature
        dispatcher.notify(event=dispatcher.posteriorStart, controller=annealer)
        # compute it
        self.cuEvalPosterior(step=step, batch=batch)
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
        self.cuVerify(step, mask, batch=step.shape[0])
        return self

    # local
    datallk = None


# end of file
