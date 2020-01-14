# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-2020 parasim inc
# (c) 2010-2020 california institute of technology
# all rights reserved
#
# Author(s): Lijun Zhu

# get the package
import altar
import altar.cuda
# get the base
from .cudaDistribution import cudaDistribution


# the declaration
class cudaPreset(cudaDistribution, family="altar.cuda.distributions.preset"):
    """
    The cuda preset distribution - initialize samples from a file
    Note that a preset distribution cannot be used for prior_run.
    """

    # user configurable state
    input_file = altar.properties.path(default=None)
    input_file.doc = "input file in hdf5 format"

    dataset = altar.properties.str(default=None)
    dataset.doc = "the name of dataset in hdf5"

    @altar.export
    def initialize(self, application):
        """
        Initialize with the given random number generator
        """

        # all done
        return self

    def cuInitialize(self, application):
        """
        cuda initialize distribution
        :param application:
        :return:
        """
        # super class process
        super().cuInitialize(application=application)

        # get information from application

        # rank is used for different thread to load different samples
        self.rank = application.controller.worker.wid
        # convert to desired precision if needed
        self.precision = application.job.gpuprecision
        # error report
        self.error = application.error
        # get the input path
        self.ifs = application.model.ifs

        # all done
        return self


    def cuInitSample(self, theta):
        """
        Fill my portion of {theta} with initial random values from my distribution.

        """

        # load from hdf5
        self._loadhdf5(theta=theta)

        # and return
        return self


    # local methods
    def _loadhdf5(self, theta):
        """
        load from hdf5 file
        """
        import h5py
        import numpy

        # grab th error channel
        channel = self.error

        # grab the input dataspace
        ifs = self.ifs
        # check the file existence
        try:
            # get the path to the file
            df = ifs[self.input_file]
        # if the file doesn't exist
        except ifs.NotFoundError:
            # complain
            channel.log(f"missing preset samples file: no '{self.input_file}' {ifs.path()}")
            # and raise the exception again
            raise

        # if all goes well

        # open file
        h5file = h5py.File(df.uri.path, 'r')
        # get the desired dataset
        if self.dataset is None:
            raise channel.log(f"missing dataset name e.g. ParameterSets/theta")
        dataset = h5file.get(self.dataset)

        # get dataset info
        dsamples, dparameters = dataset.shape

        # decide the range to copy
        # users need to check
        # 1. there are enough samples to draw
        # 2. the numbers of parameters should be the same
        samples = theta.shape[0]
        sample_start = samples * self.rank
        sample_end = sample_start + samples
        parameter_start = 0
        parameter_end = parameter_start + self.parameters

        # read the data out as a ndarray
        hmatrix = numpy.asarray(dataset[sample_start:sample_end, parameter_start:parameter_end], dtype=theta.dtype)


        # copy data to a cuda matrix
        dmatrix = altar.cuda.matrix(source=hmatrix, dtype=hmatrix.dtype)

        # copy it to the assigned position (row = 0, column = distribution/pset offset in theta)
        theta.insert(src=dmatrix, start=(0, self.offset))

        h5file.close()
        # all done
        return theta

    # local variables
    rank = 0
    precision = None
    ifs = None
    error = None

# end of file
