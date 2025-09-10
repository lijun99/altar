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
import altar.cuda
# my protocol
from altar.models.Bayesian import Bayesian

# other
import numpy

# declaration
class cudaBayesian(Bayesian, family="altar.models.cudabayesian"):
    """
    The base class of AlTar models that are compatible with Bayesian explorations
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

    psets = altar.properties.dict(schema=altar.cuda.models.parameters())
    psets.default = dict() # empty
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

    return_residual = altar.properties.bool(default=False)
    return_residual.doc = "the forward model returns residual(True) or prediction(False)"

    # options for performing forward model only
    forwardonly = altar.properties.bool(default=False)
    forwardonly.doc = "whether to run the simulation or the forward problem only"

    # input theta (one sample)
    theta_input = altar.properties.path(default="theta.h5")
    theta_input.doc = "the theta input file with a vector of parameters"

    theta_dataset = altar.properties.str(default=None)
    theta_dataset.doc = "the name/path of the theta dataset in h5 file"

    forward_output = altar.properties.path(default="forward_prediction.h5")
    forward_output.doc = "the name/path of the file to save forward problem results"

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
        # initialize the offset
        parameters = 0

        # standalone model (not embedded):
        if self.embedded is False:
            # iterate over a list of parameter sets
            for name in self.psets_list:
                # get the parameter set from psets dictionary
                pset = self.psets[name]
                # set the offset
                pset.offset = parameters
                # initialize the pset
                parameters += pset.cuInitialize(application=application)
        else: # embedded model
            # get the psets from master
            psets_master = application.model.psets
            for name in self.psets_list:
                pset = psets_master[name]
                # add to dictionary
                self.psets[name] = pset
                #self.psets.update(name=pset)
                # update parameters
                parameters += pset.count

        # the total number of parameters is now known, so record it
        self.parameters = parameters

        # set up an idx_map
        if self.idx_map is None:
            idx_map = list()
            for name, pset in self.psets.items():
                idx_map += range(pset.offset, pset.offset+pset.count)
            # sort idx in sequence
            idx_map.sort()
            self.idx_map = idx_map
        self.gidx_map = altar.cuda.vector(source=numpy.asarray(self.idx_map, dtype='int64'))

        # make a theta copy for transformation
        self.thetaPhysical = altar.cuda.matrix(shape=(self.samples, self.parameters), dtype=self.precision)

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


    def cuInitSample(self, theta, batch):
        """
        Fill {theta} with an initial random sample from my prior distribution.
        """
        # ask my subsets
        for name, pset in self.psets.items():
            # and ask each one to verify the sample
            pset.prep.cuInitSample(theta=theta, batch=batch)

        # all done
        return self

    def cuVerify(self, theta, mask, batch):
        """
        Check whether the samples in {step.theta} are consistent with the model requirements and
        update the {mask}, a vector with zeroes for valid samples and non-zero for invalid ones
        """
        # ask my subsets
        for pset in self.psets.values():
            # and ask each one to verify the sample
            pset.prior.cuVerify(theta=theta, mask=mask, batch=batch)
        # all done; return the rejection map
        return mask

    def constrain(self, theta, batch):
        """
        Check whether the samples in {step.theta} are consistent with the model requirements and
        update the {mask}, a vector with zeroes for valid samples and non-zero for invalid ones
        """
        # ask my subsets
        for pset in self.psets.values():
            # and ask each one to constrain the sample
            pset.prior.cuConstrain(theta=theta, batch=batch)
        # all done
        return self

    def cuToPhysical(self, theta, batch, inplace=False):
        """
        Transform {step.theta} to {thetaPhysical} as physical properties for the forward model
        """
        if inplace:
            thetaPhysical = theta
        else:
            # make a copy
            thetaPhysical = self.thetaPhysical
            thetaPhysical.copy(other=theta)
        # ask my subsets
        for pset in self.psets.values():
            pset.prior.cuToPhysical(theta=thetaPhysical, batch=batch)
        # all done
        return thetaPhysical

    def cuToSampling(self, theta, batch, inplace=False):
        """
        Transform {thetaPhysical} to {theta} as sampling parameters
        """
        if not inplace:
            # make a copy
            theta.copy(other=self.thetaPhysical)
        # ask my subsets
        for pset in self.psets.values():
            pset.prior.cuToSampling(theta=theta, batch=batch)
        # all done
        return theta

    def cuEvalPrior(self, theta, prior, batch):
        """
        compute the prior from the (sampling) parameter sets
        """
        # ask my subsets
        for pset in self.psets.values():
            # and ask each one to verify the sample
            pset.prior.cuEvalPrior(theta=theta, prior=prior, batch=batch)

        # all done
        return self

    def cuEvalPriorwithPhysical(self, theta, prior, batch):
        """
        Compute additional prior contributions in terms of physical parameters
        """
        # ask my subsets
        for pset in self.psets.values():
            # and ask each one to verify the sample
            pset.prior.cuEvalPriorwithPhysical(theta=theta, prior=prior, batch=batch)

        # all done
        return self


    def cuEvalPriorGradient(self, theta, prior, batch):
        """
        Fill {prior} with the log pdf gradient to {theta[index]}of the samples
        """
        # ask my subsets
        for pset in self.psets.values():
            pset.prior.cuPriorGradient(theta=theta, prior=prior, batch=batch)

        # all done
        return self

    def cuEvalPriorPhysical(self, theta, prior, batch):
        """
        Fill {priorLLK} with the log likelihoods of the samples in {theta} in my prior distribution
        """
        # ask my subsets
        for pset in self.psets.values():
            # and ask each one to verify the sample
            pset.prior.cuEvalPriorPhysical(theta=theta, prior=prior, batch=batch)

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

        batch = batch or step.samples

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
        # make theta transformation
        self.thetaPhysical = self.cuToPhysical(theta=step.theta, batch=batch)
        # compute it
        self.cuEvalPriorwithPhysical(theta=self.thetaPhysical, prior=step.prior, batch=batch)
        self.cuEvalLikelihood(theta=self.thetaPhysical, likelihood=step.data, batch=batch)
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

    def gradient(self, controller, step, parameter, batch=None):
        """
        Compute the gradient of posterior for Langevin dynamics
        step: theta
        """
        raise NotImplementedError("The model needs to define Langevin gradient method")



    @altar.export
    def verify(self, step, mask):
        """
        Check whether the samples in {step.theta} are consistent with the model requirements and
        update the {mask}, a vector with zeroes for valid samples and non-zero for invalid ones
        """
        self.cuVerify(step, mask, batch=step.samples)
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
        Load an input file to a numpy array (for both float32/64 support)
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
        :return: output numpy.array
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
                cpuData = numpy.fromfile(file.uri.path, dtype=dtype)
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
        # all done
        return cpuData


    def loadFileToGPU(self, filename, shape=None, dataset=None, out=None, dtype=None):
        """
        Load an input file to a gpu (for both float32/64 support)
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
        :return: out altar.cuda.matrix/vector
        """

        dtype = dtype or self.precision

        # load to cpu as a numpy array at fist
        cpuData = self.loadFile(filename=filename, shape=shape, dataset=dataset, dtype=dtype)

        # if output gpu matrix/vector is not pre-allocated
        if out is None:
            # if vector
            if cpuData.ndim == 1:
                out = altar.cuda.vector(shape=cpuData.shape[0], dtype=dtype)
            # if matrix
            elif cpuData.ndim == 2:
                out = altar.cuda.matrix(shape=cpuData.shape, dtype=dtype)
            else:
                channel = self.error
                raise channel.log(f"unsupported data dimension {cpuData.shape}")

        out.copy_from_host(source=cpuData)
        # all done
        return out

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

    @altar.export
    def forwardProblem(self, application, theta=None):
        """
        Perform the forward modeling with given {theta}
        """
        # each model needs to define its own version
        return

    # private data
    observations = None
    device = None
    precision = None
    ifs = None # the filesystem with the input files
    gidx_map = None # idx_map on gpu
    gtheta = None # theta with own parameters


# end of file
