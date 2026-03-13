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

# my superclass
from altar.models.Bayesian import Bayesian

# other
import numpy

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
        # chain up
        super().initialize(application=application)

        # mount my input data space
        self.ifs = self.mount_input_dataspace(pfs=application.pfs)

        # find out how many samples to work with; equal to the number of chains
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
            parameters += pset.cu_initialize(application=application)
        self.parameters = parameters

        # go through my models
        for name, model in self.models.items():
            # and initialize each one
            # set child models as embedded
            model.embedded = True
            # set child models forwardonly
            model.forwardonly = self.forwardonly
            model.initialize(application=application)

        self.cu_initialize(application=application)

        self.datallk = altar.cuda.vector(shape=self.samples, dtype=self.precision)
        # all done
        return self

    def cu_initialize(self, application):
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

    def cu_init_sample(self, theta, batch):
        """
        Fill {theta} with an initial random sample from my prior distribution.
        """
        # ask my subsets
        for name, pset in self.psets.items():
            # and ask each one to verify the sample
            pset.prep.cu_init_sample(theta=theta, batch=batch)

        # all done
        return self

    def cu_verify(self, theta, mask, batch):
        """
        Check whether the samples in {step.theta} are consistent with the model requirements and
        update the {mask}, a vector with zeroes for valid samples and non-zero for invalid ones
        """
        # ask my subsets
        for pset in self.psets.values():
            # and ask each one to verify the sample
            pset.prior.cu_verify(theta=theta, mask=mask, batch=batch)
        # all done; return the rejection map
        return mask

    def cu_eval_prior(self, theta, prior, batch):
        """
        Fill {prior} with the log likelihoods of the samples in {theta} in my prior distribution
        """
        # ask my subsets
        for pset in self.psets.values():
            # and ask each one to verify the sample
            pset.prior.cu_eval_prior(theta=theta, prior=prior, batch=batch)

        # all done
        return self

    def cu_eval_likelihood(self, step, batch):
        """
        Fill {step.data} with the likelihoods of the samples in {step.theta} given the available
        data. This is what is usually referred to as the "forward model"
        """
        datallk = self.datallk
        # ask each of my models
        for name, model in self.models.items():
            # to contribute to the computation of the data likelihood

            # each model needs to decide how it takes the whole parameter set from an ensemble
            # one option is to make a local copy of theta if needed
            # model_theta = model.restricted(theta=step.theta, batch=batch)
            # another is to use idx_map

            model.cu_eval_likelihood(theta=step.theta, likelihood=datallk.zero(), batch=batch)
            if model.cascaded:
                step.prior += datallk
            else:
                step.data += datallk

        # all done
        return self

    def cu_eval_posterior(self, step, batch):
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

    def update_model(self, annealer):
        """
        Update model parameters if needed
        :param annealer:
        :return:
        """
        # default is not updated
        out = False
        # iterate over embedded models
        for name, model in self.models.items():
            updated = model.update_model(annealer=annealer)
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
        dispatcher.notify(event=dispatcher.prior_start, controller=annealer)
        # compute the prior likelihood
        self.cu_eval_prior(theta=step.theta, prior=step.prior, batch=batch)
        # done
        dispatcher.notify(event=dispatcher.prior_finish, controller=annealer)

        # notify we are about to compute the likelihood of the prior given the data
        dispatcher.notify(event=dispatcher.data_start, controller=annealer)
        # compute it
        self.cu_eval_likelihood(step=step, batch=batch)
        # done
        dispatcher.notify(event=dispatcher.data_finish, controller=annealer)

        # finally, notify we are about to put together the posterior at this temperature
        dispatcher.notify(event=dispatcher.posterior_start, controller=annealer)
        # compute it
        self.cu_eval_posterior(step=step, batch=batch)
        # done
        dispatcher.notify(event=dispatcher.posterior_finish, controller=annealer)

        # enable chaining
        return self


    @altar.export
    def verify(self, step, mask):
        """
        Check whether the samples in {step.theta} are consistent with the model requirements and
        update the {mask}, a vector with zeroes for valid samples and non-zero for invalid ones
        """
        self.cu_verify(step, mask, batch=step.shape[0])
        return self

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

    def load_file(self, filename, shape=None, dataset=None, dtype=None):
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

    def load_file_to_gpu(self, filename, shape=None, dataset=None, out=None, dtype=None):
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
        cpuData = self.load_file(filename=filename, shape=shape, dataset=dataset, dtype=dtype)

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

    @altar.export
    def forward_problem(self, application, theta=None):
        """
        Perform the forward modeling with given {theta}
        """
        # only for one set of parameters
        theta = theta or self.load_file_to_gpu(filename=self.theta_input,
                                            dataset=self.theta_dataset)
        for name, model in self.models.items():
            # to contribute to the computation of the data likelihood

            # use the ensemble output path if not provided for each model
            # model needs to decide how to treat the whole parameter set

            model.forward_output = self.forward_output
            model.forward_problem(application=application, theta=theta)
        return

    # local
    datallk = None


# end of file
