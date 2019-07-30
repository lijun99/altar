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
class cudaKinematicG(cudaBayesian, family="altar.models.seismic.cuda.kinematicg"):
    """
    KinematicG model with cuda
    """

    # configurable traits

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

    # public data
    cmodel = None

    # protocol obligations
    @altar.export
    def initialize(self, application):
        """
        Initialize the state of the model given a {problem} specification
        """
        # chain up
        super().initialize(application=application)

        # get a cublas handle
        self.cublas_handle = self.device.get_cublas_handle()

        # load the green's function
        self.NGbparameters = 2*self.Nas*self.Ndd*self.Nt
        self.gGF=self.loadFileToGPU(filename=self.green, shape=(self.NGbparameters, self.observations))
        # merge covariance to gf
        self.mergeCovarianceToGF()

        # prepare the residuals matrix
        self.gDprediction = altar.cuda.matrix(shape=(self.samples, self.observations), dtype=self.precision)

        # prepare the initial arrival time
        self.gt0s = altar.cuda.vector(source=numpy.asarray(self.t0s, dtype=self.precision))

        # create a cuda/c model object
        dtype = self.gGF.dtype.num
        self.cmodel = libcudaseismic.kinematicg_alloc(
                self.Nas, self.Ndd, self.Nmesh, self.dsp,
                self.Nt, self.Npt, self.dt,
                self.gt0s.data,
                self.samples, self.parameters, self.observations,
                self.gidx_map.data, dtype)

        # all done
        return self

    def forwardModelBatched(self, theta, gf, prediction, batch, observation=None):
        """
        KinematicG forward model in batch: cast Mb(x,y,t)
        :param theta: matrix (samples, parameters), sampling parameters
        :param gf: matrix (2*Ndd*Nas*Nt, observations), kinematicG green's function
        :param prediction: matrix (samples, observations), the predicted data or residual between predicted and observed data
        :param batch: integer, the number of samples to be computed batch<=samples
        :param observation: matrix (samples, observations), duplicates of observed data
        :return: prediction as predicted data(observation=None) or residual (observation is provided)
        """
        if observation is None:
            return_residual = False
        else:
            prediction.copy(other=observation)
            return_residual = True

        # call cuda/c library
        libcudaseismic.kinematicg_forward_batched(self.cublas_handle, self.cmodel,
            theta.data, gf.data, prediction.data, theta.shape[1], batch, return_residual)

        # all done
        return prediction

    def forwardModel(self, theta, gf, prediction, observation=None):
        """
        KinematicG forward model for single sample: cast Mb(x,y,t)
        :param theta: vector (parameters), sampling parameters
        :param gf: matrix (2*Ndd*Nas*Nt, observations), kinematicG green's function
        :param prediction: vector (observations), the predicted data or residual between predicted and observed data
        :param observation: vector (observations), duplicates of observed data
        :return: prediction as predicted data(observation=None) or residual (observation is provided)
        """
        if observation is None:
            return_residual = False
        else:
            prediction.copy(other=observation)
            return_residual = True

        parameters = theta.shape
        # call cuda/c extension
        libcudaseismic.kinematicg_forward(self.cublas_handle, self.cmodel,
            theta.data, gf.data, prediction.data, parameters, return_residual)

        # all done
        return prediction


    def cuEvalLikelihood(self, theta, likelihood, batch):
        """
        to be loaded by super class cuEvalLikelihood which already decides where the local likelihood is added to
        """
        residuals = self.gDprediction
        # call forward to caculate the data prediction or its difference between dataobs
        self.forwardModelBatched(theta=theta, gf=self.gGF, prediction=residuals, batch=batch,
                observation= self.dataobs.gdataObsBatch)
        # call data to calculate the l2 norm
        self.dataobs.cuEvalLikelihood(prediction=residuals, likelihood=likelihood,
            residual=True, batch=batch)
        # return the likelihood
        return likelihood

    def mergeCovarianceToGF(self):
        """
        merge cd with green function
        """

        # merge cd with Green's function
        cd_inv = self.dataobs.gcd_inv
        green = self.gGF
        # check whether cd is a constant or a matrix
        if isinstance(cd_inv, float):
            green *= cd_inv
        elif isinstance(cd_inv, altar.cuda.matrix):
            # (NGbparameters x obs) x (obsxobs)  = (NGbparameters x obs)
            cublas.trmm(cd_inv, green, out=green, side=cublas.SideRight, uplo=cublas.FillModeUpper,
                transa = cublas.OpNoTrans, diag=cublas.DiagNonUnit, alpha=1.0,
                handle = self.cublas_handle)


        # all done
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
