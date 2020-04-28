# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-2020 parasim inc
# (c) 2010-2020 california institute of technology
# all rights reserved
#
# Author(s): Lijun Zhu


# the package
import altar
# my protocol
from .Bayesian import Bayesian

# other
import numpy

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

    psets_list = altar.properties.list(default=None)
    psets_list.doc = "list of parameter sets, used to set orders"

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

        # mount my input data space
        self.ifs = self.mountInputDataspace(pfs=application.pfs)
        # find out how many samples I will be working with; this equal to the number of chains
        self.samples = application.job.chains

        # initialize the data
        self.dataobs.initialize(application=application)
        self.observations = self.dataobs.observations

        # initialize the parametersets
        # initialize the offset
        psets = self.psets
        # initialize the offset
        offset = 0

        for name in self.psets_list:
            # get the parameter set from psets dictionary
            pset = self.psets[name]
            # initialize the parameter set
            offset += pset.initialize(model=self, offset=offset)
        # the total number of parameters is now known, so record it
        self.parameters = offset

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
    def initializeSample(self, step):
        """
        Fill {step.θ} with an initial random sample from my prior distribution.
        """
        # grab the portion of the sample that's mine
        θ = self.restrict(theta=step.theta)
        # go through each parameter set
        for pset in self.psets.values():
            # and ask each one to {prep} the sample
            pset.initializeSample(theta=θ)
        # and return
        return self

    @altar.export
    def verify(self, step, mask):
        """
        Check whether the samples in {step.theta} are consistent with the model requirements and
        update the {mask}, a vector with zeroes for valid samples and non-zero for invalid ones
        """
        # grab the portion of the sample that's mine
        θ = self.restrict(theta=step.theta)
        # ask my subsets
        for pset in self.psets.values():
            # and ask each one to verify the sample
            pset.verify(theta=θ, mask=mask)
        # all done; return the rejection map
        return mask

    def evalPrior(self, theta, prior):
        """
        Fill {priorLLK} with the log likelihoods of the samples in {theta} in my prior distribution
        """
        # ask my subsets
        for pset in self.psets.values():
            # and ask each one to verify the sample
            pset.priorLikelihood(theta, prior)

        # all done
        return self


    def forwardModel(self, theta, prediction):
        """
        The forward model for a single set of parameters
        """
        # i don't know what to do, so...
        raise NotImplementedError(
            f"model '{type(self).__name__}' must implement 'forwardModel'")


    def forwardModelBatched(self, theta, prediction):
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
            self.forwardModel(theta=theta_sample, prediction=prediction_sample)
            # copy to the prediction matrix
            prediction.setRow(sample, prediction_sample)

        # all done
        return self


    def evalDataLikelihood(self, theta, likelihood):
        """
        calculate data likelihood and add it to step.prior or step.data
        """
        # This method assumes that there is a forwardModelBatched defined
        # Otherwise, please define your own version of this method

        # create a matrix for the prediction (samples, observations)
        prediction = altar.matrix(shape=(self.samples, self.observations))
        # survey forward model whether it computes residual or not
        returnResidual = self.return_residual
        # call forwardModel to calculate the data prediction or its difference between dataobs
        self.forwardModelBatched(theta=theta, prediction=prediction)
        # call data to calculate the l2 norm
        self.dataobs.evalLikelihood(prediction=prediction, likelihood=likelihood, residual=returnResidual)

        # all done
        return self


    def evalPosterior(self, step):
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

        batch = step.samples

        # grab the dispatcher
        dispatcher = annealer.dispatcher

        # notify we are about to compute the prior likelihood
        dispatcher.notify(event=dispatcher.priorStart, controller=annealer)
        # compute the prior likelihood
        self.evalPrior(theta=step.theta, prior=step.prior)
        # done
        dispatcher.notify(event=dispatcher.priorFinish, controller=annealer)

        # notify we are about to compute the likelihood of the prior given the data
        dispatcher.notify(event=dispatcher.dataStart, controller=annealer)

        # grab the portion of the sample that's mine
        θ = self.restrict(theta=step.theta)
        # compute it
        self.evalDataLikelihood(theta=θ, likelihood=step.data)
        # done
        dispatcher.notify(event=dispatcher.dataFinish, controller=annealer)

        # finally, notify we are about to put together the posterior at this temperature
        dispatcher.notify(event=dispatcher.posteriorStart, controller=annealer)
        # compute it
        self.evalPosterior(step=step)
        # done
        dispatcher.notify(event=dispatcher.posteriorFinish, controller=annealer)

        # enable chaining
        return self


    def updateModel(self, annealer):
        """
        Update Model parameters if needed
        :param annealer:
        :return: default is False
        """
        return False


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

    def loadFile(self, filename, shape=None, dataset=None, dtype=None):
        """
        Load an input file to a gsl vector or matrix (for both float32/64 support)
        Supported format:
        1. text file in '.txt' suffix, stored in prescribed shape
        2. binary file with '.bin' or '.dat' suffix,
            the precision must be same as the desired gpuprecision,
            and users must specify the shape of the data
        3. (preferred) hdf5 file in '.h5' suffix (preferred)
            the metadata of shape, precision is included in .h5 file
        :param filename: str, the input file name
        :param shape: list of int
        :param dataset: str, name/key of dataset for h5 input only
        :return: output gsl vector/matrix
        """

        # decide the data type of the loaded vector/matrix
        dtype = dtype or self.precision

        ifs = self.ifs
        channel = self.error
        try:
            # get the path to the file
            file = ifs[filename]
        except not ifs.NotFoundError:
            channel.log(f"no file '{filename}' found in '{ifs.path()}'")
            raise
        else:
            # get the suffix to determine type
            suffix = file.uri.suffix
            # use .txt for non-binary input
            if suffix == '.txt':
                # load to a cpu array
                cpuData = numpy.loadtxt(file.uri.path, dtype=dtype)
            # binary data
            elif suffix == '.bin' or suffix == '.dat':
                # check shape
                if shape is None:
                    # check whether I can get shape from output
                    if out is None:
                        raise channel.log(f"must specify shape for binary input '{filename}'")
                    else:
                        shape = out.shape
                # read and reshape, users need to check the precision
                cpuData = numpy.fromfile(file.uri.path, dtype=self.precision).reshape(shape)
            # hdf5 file
            elif suffix == '.h5':
                # get support
                import h5py
                # open
                h5file = h5py.File(file.uri.path, 'r')
                # get the desired dataset
                if dataset is None:
                    # if not provided, assume the only or first dataset as default
                    dataset = list(h5file.keys())[0]
                cpuData = numpy.asarray(h5file.get(dataset), dtype=dtype)
                h5file.close()

        if shape is not None:
            cpuData = cpuData.reshape(shape)

        # convert to gsl data


        # all done
        return cpuData

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

    # private data
    observations = None
    device = None
    precision = None
    ifs = None # the filesystem with the input files


# end of file
