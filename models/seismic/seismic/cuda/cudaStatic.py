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
from altar.cuda import cublas
from altar.cuda import libcuda
from altar.cuda.models.cudaBayesian import cudaBayesian

# declaration
class cudaStatic(cudaBayesian, family="altar.models.seismic.cudastatic"):
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
        self.GF = self.loadGF()
        self.prepareGF()

        # prepare the residuals matrix
        self.gDprediction = altar.cuda.matrix(shape=(self.samples, self.observations), dtype=self.precision)
        # all done
        return self

    def _forwardModel(self, theta, prediction, batch, observation=None):
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
        # in c/python pred (samplesxobs), green (obsxparameters), theta (samplesxparams)  
        # translated to cublas/fortran, pred (obsxsamples) green(param obs) theta (params x samples)
        # we therefore use pred = G^T x theta
        green = self.gGF
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
            green = altar.matrix(shape=(self.observations, self.parameters))
            # and load the file contents into memory
            green.load(gf.uri)
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
        # (obsxobs) x (obsxparameters) = (obsxparameters)
        cublas.trmm(cd_inv, green, out=green, side=cublas.SideLeft, uplo=cublas.FillModeUpper,
            transa = cublas.OpNoTrans, diag=cublas.DiagNonUnit, alpha=1.0,
            handle = self.cublas_handle)
        #cublas.gemm(cd_inv, green, out=green)
        # all done
        return


    # private data
    # inputs
    GF = None # the Green functions
    gGF = None
    gDprediction = None
    cublas_handle=None

# end of file
