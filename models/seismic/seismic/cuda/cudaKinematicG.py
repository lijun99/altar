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
# my base 
from altar.cuda.models.cudaBayesian import cudaBayesian
# extensions
from altar.cuda import cublas
from altar.cuda import libcuda
from altar.models.seismic.ext import cudaseismic as libcudaseismic
import numpy

# declaration
class cudaKinematicG(cudaBayesian, family="altar.models.seismic.cudakinematicg"):
    """
    cudaLinear with the new cuda framework
    """

    # data observations
    dataobs = altar.cuda.data.data()
    dataobs.default = altar.cuda.data.datal2()
    dataobs.doc = "the observed data"

    # the file based inputs
    green = altar.properties.path(default="green.txt")
    green.doc = "the name of the file with the Green functions"

    Nas = altar.properties.int(default=1)
    Nas.doc = "number of patches along strike direction"

    Ndd = altar.properties.int(default=1)
    Ndd.doc = "number of patches along dip direction"

    Nmesh = altar.properties.int(default=1)
    Nmesh.doc = "number of mesh points for each patch for fastsweeping"

    dsp = altar.properties.float(default=10.0)
    dsp.doc = "the distance unit for each patch, in km"

    Nt = altar.properties.int(default=1)
    Nt.doc = "number of time intervals for kinematic process"

    Npt = altar.properties.int(default=1)
    Npt.doc = "number of mesh points for each time interval for fastsweeping"

    dt = altar.properties.float(default=1.0)
    dt.doc = "the time unit for each time interval (in s)"

    t0s = altar.properties.array(default=None)
    t0s.doc = "the start time for each patch"
    
    cmodel = None

    # protocol obligations
    @altar.export
    def initialize(self, application):
        """
        Initialize the state of the model given a {problem} specification
        """
        super().initialize(application=application)
        # chain up
        self.cublas_handle = self.device.get_cublas_handle()
        # convert the input filenames into data
        self.NGbparameters = 2*self.Nas*self.Ndd*self.Nt
        self.GF=self.loadGF()
        self.prepareGF()

        # prepare the residuals matrix
        self.gDprediction = altar.cuda.matrix(shape=(self.samples, self.observations), dtype=self.precision)

        self.gt0s = altar.cuda.vector(source=numpy.asarray(self.t0s, dtype=self.precision))
        
        dtype = self.gGF.dtype.num
        # create a cuda/c model
        self.cmodel = libcudaseismic.kinematicg_alloc(
                self.Nas, self.Ndd, self.Nmesh, self.dsp, 
                self.Nt, self.Npt, self.dt,
                self.gt0s.data, 
                self.samples, self.parameters, self.observations,
                self.gidx_map.data, dtype)
        
        # all done
        return self

    def _forwardModel(self, theta, prediction, batch, observation=None):
        """
        KinematicG forward model: cast Mb(x,y,t)    
        """
        if observation is None:
            return_residual = False
        else:
            prediction.copy(other=observation)
            return_residual = True

        # call cuda/c library
        libcudaseismic.kinematicg_forward(self.cublas_handle, self.cmodel,
            theta.data, self.gGF.data, prediction.data, theta.shape[1], batch, return_residual)

        
        # all done
        return self
        
    
    def cuEvalLikelihood(self, theta, likelihood, batch):
        """
        to be loaded by super class cuEvalLikelihood which already decides where the local likelihood is added to
        """
        residuals = self.gDprediction
        # call forward to caculate the data prediction or its difference between dataobs
        self._forwardModel(theta=theta, prediction=residuals, batch=batch,
                observation= self.dataobs.gdataObsBatch)  
        # call data to calculate the l2 norm
        self.dataobs.cuEvalLikelihood(prediction=residuals, likelihood=likelihood,
            residual=True, batch=batch)
        # return the likelihood        
        return likelihood


    def loadGF(self):
        """
        Load the data in the input files into memory
        """
        # grab the input dataspace
        ifs = self.ifs

        # first the green functions
        try:
            # get the path to the file
            gf = ifs[self.green]
        # if the file doesn't exist
        except ifs.NotFoundError:
            # grab my error channel
            channel = self.error
            # complain
            channel.log(f"missing Green functions: no '{self.green}' in '{self.case}'")
            # and raise the exception again
            raise
        # if all goes well
        else:
            # allocate the matrix
            green = altar.matrix(shape=(self.NGbparameters, self.observations))
            # and load the file contents into memory
            green.load(gf.uri, binary=True)
        # all done
        return green

    def prepareGF(self):
        """
        copy green function to gpu and merge cd with green function
        """
        # make a gpu copy
        self.gGF = altar.cuda.matrix(source=self.GF, dtype=self.precision)
        # merge cd with Green's function
        cd_inv = self.dataobs.gcd_inv
        green = self.gGF
        if isinstance(cd_inv, float):
            green *= cd_inv
        elif isinstance(cd_inv, altar.matrix):
            # (obsxobs) x (obsxparameters) = (obsxparameters)
            cublas.trmm(cd_inv, green, out=green, side=cublas.SideLeft, uplo=cublas.FillModeUpper,
                transa = cublas.OpNoTrans, diag=cublas.DiagNonUnit, alpha=1.0,
                handle = self.cublas_handle)
        #cublas.gemm(cd_inv, green, out=green)
        return


    # private data
    # inputs
    GF = None # the Green functions
    gGF = None
    gDprediction = None
    cublas_handle=None
    NGbparameters = None
    gt0s = None

# end of file
