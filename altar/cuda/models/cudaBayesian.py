# -*- python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# Lijun Zhu <ljzhu@gps.caltech.edu>
#
# (c) 2013-2019 parasim inc
# (c) 2010-2019 california institute of technology
# all rights reserved
#


# the package
import altar
import altar.cuda
# my protocol
from altar.models.Bayesian import Bayesian

# other
import numpy

# declaration
class cudaBayesian(Bayesian, family="altar.cuda.models.bayesian"):
    """
    The base class of AlTar models that are compatible with Bayesian explorations
    """

    # user configurable state

    parameters = altar.properties.int(default=1)
    parameters.doc = "the number of model degrees of freedom"

    cascaded = altar.properties.bool(default=False)
    cascaded.doc = "whether the model is cascaded, aka, annealing temperature always = 1"    

    embedded = altar.properties.bool(default=False)
    embedded.doc = "whether the model is embedded in an ensemble of models"

    psets = altar.properties.dict(schema=altar.cuda.models.parameters())
    psets.doc = "an ensemble of parameter sets in the model"

    dataobs = altar.cuda.data.data()
    dataobs.default = altar.cuda.data.datal2()
    dataobs.doc = "observed data"

    # the path of input files
    case = altar.properties.path(default="input")
    case.doc = "the directory with the input files"

    idx_map=altar.properties.list(schema=altar.properties.int())
    idx_map.default = None
    idx_map.doc = "the indices for model parameters in whole theta set"
    

    # protocol obligations
    @altar.export
    def initialize(self, application):
        """
        Initialize the state of the model given an {application} context
        """
        # super class method
        super().initialize(application=application)

        # mount my input data space
        self.ifs = self.mountInputDataspace(pfs=application.pfs)
        # find out how many samples I will be working with; this equal to the number of chains
        self.samples = application.job.chains


        # cuda method
        self.device = application.controller.worker.device
        self.precision = application.job.gpuprecision

        # initialize the data
        self.dataobs.initialize(application=application)
        self.observations = self.dataobs.observations

        # initialize the parametersets
        psets = self.psets
        # initialize the offset
        parameters = 0
        # go through my parameter sets
        for name, pset in psets.items():
            # initialize the parameter set
            if self.embedded:
                parameters += pset.count
            else:
                parameters += pset.cuInitialize(application=application)

            #print("name", name, pset.offset, pset.count, parameters)
        # the total number of parameters is now known, so record it
        self.parameters = parameters
        
        # set up an idx_map
        if self.idx_map is None:
            idx_map = list()
            for name, pset in psets.items():
                idx_map += range(pset.offset, pset.offset+pset.count)
            # sort idx in sequence   
            idx_map.sort()
            self.idx_map = idx_map 
        self.gidx_map = altar.cuda.vector(source=numpy.asarray(self.idx_map, dtype='int64'))
        
        # all done
        return self

    def cuInitialize(self, application):
        """
        cuda interface
        """

        #all done
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
        # ask my subsets
        for pset in self.psets.values():
            # and ask each one to verify the sample
            pset.prior.cuEvalPrior(theta=theta, prior=prior, batch=batch)

        # all done
        return self


    def cuEvalLikelihood(self, theta, likelihood, batch):
        """
        calculate data likelihood and add it to step.prior or step.data
        """
        # model has to define this
        return self


    def cuEvalPosterior(self, step, batch):
        """
        Given the {step.prior} and {step.data} likelihoods, compute a generalized posterior using
        {step.beta} and deposit the result in {step.post}
        """
        batch = batch if batch is not None else step.samples
        # prime the posterior
        step.posterior.copy(step.prior)
        # compute it; this expression reduces to Bayes' theorem for β->1
        altar.cuda.cublas.axpy(alpha=step.beta, x=step.data, y=step.posterior, batch=batch)
        # all done
        return self


    @altar.export
    def likelihoods(self, annealer, step, batch=None):
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
        self.cuEvalLikelihood(theta=step.theta, likelihood=step.data, batch=batch)
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
        self.cuVerify(step, mask, batch=step.samples)
        return self


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

    def restricted(self, theta, batch):
        """
        extract theta which contains model's own parameters
        """
        # allocate model's theta on gpu
        if self.gtheta is None:
            self.gtheta = altar.cuda.matrix(shape=(theta.shape[0], self.parameters), dtype=theta.dtype)

        # extract theta according to idx_map
        theta.copycols(dst=self.gtheta, indices=self.gidx_map, batch=batch)
        # all done 
        return self.gtheta
            
    # private data
    observations = None
    device = None
    precision = None
    ifs = None # the filesystem with the input files
    gidx_map = None # idx_map on gpu
    gtheta = None # theta with own parameters


# end of file
