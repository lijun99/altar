# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-2021 parasim inc
# (c) 2010-2021 california institute of technology
# all rights reserved
#

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
class cudaLinearViscous(cudaBayesian, family="altar.models.seas.cuda.linearviscous"):
    """
    Creep model with linear viscous rheology
    """

    # configurable traits

    # data observations
    dataobs = altar.cuda.data.data()
    dataobs.default = altar.cuda.data.datal2()
    dataobs.doc = "the observed data"

    # the file based inputs
    green = altar.properties.path(default="green.txt")
    green.doc = "the name of the file with the Green functions"

    patches = altar.properties.int(default=1)
    patches.doc = "number of patches"

    stresskernel = altar.properties.path(default="stresskernel.txt")
    stresskernel.doc = "the filename for input stress kernel - patches x patches matrix"

    t = altar.properties.array(default=(0.0, 1.0))
    t.doc = "the start/end time"

    # public data
    ode_solver = None

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
        self.GF=self.loadFile(filename=self.green, shape=(self.NGbparameters, self.observations))

        # prepare the GF in gpu
        self.gGF = altar.cuda.matrix(shape=self.GF.shape, dtype=self.precision)

        # merge covariance to gf
        if not self.forwardonly:
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
        Compute the likelihood from my forward problem

        """

        # residuals = dataPrediction - dataObservation
        residuals = self.gDprediction

        # call forward model to calculate the data prediction or its difference between dataobs
        self.forwardModelBatched(theta=theta, gf=self.gGF, prediction=residuals, batch=batch,
                observation= self.dataobs.gdataObsBatch)

        # call data method to calculate the l2 norm
        self.dataobs.cuEvalLikelihood(prediction=residuals, likelihood=likelihood,
            residual=True, batch=batch)
        # return the likelihood
        return likelihood


    @altar.export
    def forwardProblem(self, application, theta=None):
        """
        Perform the forward modeling with given {theta}
        """

        # all done
        return


    # private data
    # inputs
    GF = None # the Green functions
    gGF = None
    gDprediction = None
    cublas_handle=None

# end of file
