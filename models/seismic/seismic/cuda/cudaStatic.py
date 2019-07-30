# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-2019 parasim inc
# (c) 2010-2019 california institute of technology
# all rights reserved
#
# Author(s): Lijun Zhu


# This is a copy of cudaLinear
# If used in cascaded model, make sure its parameters are the contiguous parameters in the beginning
# Otherwise, modifications of the following code are needed
# An easy way is to use self.restricted method to extract own parameters from theta

# the package
import altar
import altar.cuda
from altar.cuda import cublas as cublas
from altar.cuda import libcuda
from altar.cuda.models.cudaBayesian import cudaBayesian
import numpy

# declaration
class cudaStatic(cudaBayesian, family="altar.models.seismic.cuda.static"):
    """
    cudaLinear with the new cuda framework
    """

    # data observations
    dataobs = altar.cuda.data.data()
    dataobs.default = altar.cuda.data.datal2()
    dataobs.doc = "the observed data"

    # the file based inputs
    green = altar.properties.path(default="static.gf.h5")
    green.doc = "the name of the file with the Green functions"

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

        # load green's function to CPU
        self.GF = self.loadFile(filename=self.green, shape=(self.observations, self.parameters))
        # make a gpu copy of Green's function
        self.gGF = altar.cuda.matrix(shape=self.GF.shape, dtype=self.precision)
        # prepare the residuals matrix
        self.gDataPred = altar.cuda.matrix(shape=(self.samples, self.observations),
                                           dtype=self.precision)
        # merge covariance to green's function
        self.mergeCovarianceToGF()

        # all done
        return self

    def forwardModelBatched(self, theta, green, prediction, batch, observation=None):
        """
        Linear Forward Model prediction= G theta
        """
        # whether data observation is provided
        # gemm C = alpha A B + beta C
        if observation is None:
            beta = 0.0
        else:
            # make a copy
            prediction.copy(observation)
            # to be subtracted in gemm
            beta = -1.0

        # forward model
        # prediction = Green * theta
        # in c/python pred (samples, obs), green (obs, parameters), theta (samples, params)

        # cublas.gemm(A=theta, B=green, transa=0, transb=1,
        #                out=prediction,
        #                handle=self.cublas_handle,
        #                alpha=1.0, beta=beta,
        #                rows=batch)

        # use cublas interface directly, as in cascaded problem, only the first few parameters are used
        # in column major: translated to pred (obsxsamples) green(param obs) theta (params x samples)
        # we therefore use pred = G^T x theta

        libcuda.cublas_gemm(self.cublas_handle,
                            1, 0, # transa, transb
                            prediction.shape[1], batch, green.shape[1], # m, n, k
                            1.0,   # alpha
                            green.data, green.shape[1], # A, lda
                            theta.data, theta.shape[1], # B, ldb
                            beta,
                            prediction.data, prediction.shape[1])

        # all done
        return self


    def forwardModel(self, theta, green, prediction, observation=None):
        """
        Static/Linear forward model prediction = green * theta
        :param theta: a parameter set, vector with size parameters
        :param green: green's function, matrix with size (observations, parameters)
        :param prediction: data prediction, vector with size observations
        :return: data prediction if observation is none; otherwise return residual
        """

        if observation is None:
            beta = 0.0
        else:
            # make a copy
            prediction.copy(observation)
            # to be subtracted in gemm
            beta = -1.0

        # green (obs, params) theta (params) predic(obs)
        # cublas.gemv(handle=self.cublas_handle,
        #            A=green, trans=cublas.OpNoTrans, x=theta,
        #            out=prediction,
        #            alpha=1.0, beta=beta)

        # cublas uses column major, geenf is treated as  (params, obs)
        libcuda.cublas_gemv(self.cublas_handle,
                            1, # transa = transpose
                            green.shape[1], green.shape[0], # m, n, or param, obs
                            1.0, # alpha
                            green.data, green.shape[1], # A, lda
                            theta.data, 1, # x, incx
                            beta, # beta
                            prediction.data, 1 # y, incy
                            )

        # all done
        return self


    def cuEvalLikelihood(self, theta, likelihood, batch):
        """
        Compute data likelihood from the forward model,
        :param theta: parameters, matrix [samples, parameters]
        :param likelihood: data likelihood P(d|theta), vector [samples]
        :param batch: the number of samples to be computed, batch <=samples
        :return: likelihood, in case of model ensembles, data likelihood of this model
        is added to the input likelihood
        """

        residuals = self.gDataPred
        # call forward to caculate the data prediction or its difference between dataobs
        self.forwardModelBatched(theta=theta, green=self.gGF,
                                 prediction=residuals, batch=batch,
                                 observation= self.dataobs.gdataObsBatch)
        # compute the data likelihood with l2 norm
        self.dataobs.cuEvalLikelihood(prediction=residuals,
                                      likelihood=likelihood,
                                      residual=True, batch=batch)

        # return the likelihood
        return likelihood

    def mergeCovarianceToGF(self):
        """
        merge data covariance (cd) with green function
        """
        # get references for data covariance
        cd_inv = self.dataobs.gcd_inv
        # get a reference for green's function
        green = self.gGF
        # copy from CPU
        green.copy_from_host(source=self.GF)
        # check whether cd is a constant or a matrix
        if isinstance(cd_inv, float):
            green *= cd_inv
        elif isinstance(cd_inv, altar.cuda.matrix):
            # (obsxobs) x (obsxparameters) = (obsxparameters)
            cublas.trmm(cd_inv, green, out=green, side=cublas.SideLeft,
                        uplo=cublas.FillModeUpper,
                        transa = cublas.OpNoTrans,
                        diag=cublas.DiagNonUnit,
                        alpha=1.0,
                        handle = self.cublas_handle)
        # all done
        return

    # private data
    # inputs
    GF = None # the Green functions
    gGF = None
    gDataPred = None
    cublas_handle=None

# end of file
