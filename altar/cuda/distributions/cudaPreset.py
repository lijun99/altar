# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-2019 parasim inc
# (c) 2010-2019 california institute of technology
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
    """
    
    # user configurable state
    input_file =  altar.properties.path(default=None)
    input_file.doc = "input file in hdf5 (other format to be implemented)"  
    input_offset = altar.properties.int(default=0)
    input_offset.doc = "the offset of parameters in the preset theta set"
    
    @altar.export
    def initialize(self, application):
        """
        Initialize with the given random number generator
        """
        # super class process 
        super().initialize(application=application)

        # get worker id
        # different rank loads different samples from input
        self.rank = application.controller.worker.wid

        # all done
        return self
    
    def cuInitSample(self, theta):
        """
        Fill my portion of {theta} with initial random values from my distribution.
        """
        # batch = theta.shape[0]
        # load from hdf5 
        self._loadhdf5(theta=theta)
        
        # other format to be done
        # can use numpy functions to load to a numpy array, then insert to theta
        # see loadhdf5 for examples 
    
        # and return
        return self

    # Preset distribution cannot be used for prior_run

    # local methods
    def _loadhdf5(self, theta):
        """
        load 
        """
        import h5py
        import numpy

        # open file
        h5file = h5py.File(self.input_file, 'r')
        # create a dataset from file
        dataset = h5file.get('theta')
        # get dataset info
        dsamples, dparameters = dataset.shape

        # decide the range to copy
        # users need to check
        # 1. there are enough samples to draw
        # 2. the numbers of parameters should be the same
        samples = theta.shape[0]
        sample_start = samples * self.rank
        sample_end = sample_start + samples
        parameter_start = self.input_offset
        parameter_end = parameter_start + self.parameters
            
        # read the data out as a ndarray
        hmatrix = numpy.array(dataset[sample_start:sample_end, parameter_start:parameter_end], dtype=theta.dtype)
        h5file.close()

        # create a cuda matrix
        dmatrix = altar.cuda.matrix(source=hmatrix, dtype=hmatrix.dtype)
        # copy it to the assigned position
        theta.insert(src=dmatrix, start=(0,self.idx_range[0]))
        return theta
    
    # local variables
    rank = 0


# end of file
