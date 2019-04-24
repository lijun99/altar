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
from altar.cuda import libcuda
from altar.cuda import cublas


# my protocol
from altar.data.DataL2 import DataL2



# declaration
class cudaDataL2(DataL2, family="altar.data.cudadatal2"):
    """
    The observed data with L2 norm
    """
    # configuration states from cpu counterpart
    # data_file = altar.properties.path(default="data.txt")
    # observations = altar.properties.int(default=1)
    # cd_file = altar.properties.path(default=None)
    # cd_std = altar.properties.float(default=1.0)

    data_file = altar.properties.path(default="data.txt")
    data_file.doc = "the name of the file with the observations"

    observations = altar.properties.int(default=1)
    observations.doc = "the number of observed data"

    cd_file = altar.properties.path(default=None)
    cd_file.doc = "the name of the file with the data covariance matrix"
    
    cd_std = altar.properties.float(default=1.0)
    cd_std.doc = "the constant covariance for data, sigma^2"

    # the norm to use for computing the data log likelihood
    norm = altar.norms.norm()
    norm.default = altar.cuda.norms.l2()
    norm.doc = "the norm used to compute likelihood"

    #ifs = None

    @altar.export
    def initialize(self, application):
        """
        Initialize data obs from model
        """
        # load the gpu part at first
        # initCovariance depends on precision
        self.device = application.controller.worker.device
        self.precision = application.job.gpuprecision


        # get the input path from model
        self.ifs = application.pfs["inputs"]
        self.error = application.error
        # get the number of samples
        self.samples = application.job.chains
        # load the data and covariance
        self.loadData()
        # compute inverse of covariance, normalization
        self.initializeCovariance()
        # all done
        return self


    def cuEvalLikelihood(self, prediction, likelihood, residual=True, batch=None):
        """
        compute the datalikelihood for prediction (samples x observations)
        """

        # depending on convenience, users can
        # either copy dataobs to their model and use the residual as input of prediction
        # 
        # or compute prediction from forward model and subtract the dataobs here

        # subtract dataobs from prediction to get residualt
        if not residual:
            prediction -= self.gdataObsBatch

        # call L2 norm to calculate the likelihood
        likelihood = self.norm.cuEvalLikelihood(data=prediction, constant=self.normalization,
            out=likelihood, batch=batch)
   
        # all done
        return likelihood

    #def loadData(self):
    #use cpu routine to loadData

    @property
    def cd_inv(self):
        """
        Inverse of data covariance, in Cholesky decomposed form
        """
        return self.gcd_inv
        
    @property
    def dataobsBatch(self):
        """
        A batch of duplicated observations
        """
        return self.gdataObsBatch
        
        
    def loadData(self):
        """
        load data and covariance
        """
        
        # grab the input dataspace
        ifs = self.ifs
        # next, the observations
        try:
            # get the path to the file
            df = ifs[self.data_file]
        # if the file doesn't exist
        except ifs.NotFoundError:
            # grab my error channel
            channel = self.error
            # complain
            channel.log(f"missing observations: no '{self.data_file}' {ifs.path()}")
            # and raise the exception again
            raise
        # if all goes well
        else:
            # allocate the vector
            self.dataobs= altar.vector(shape=self.observations)
            # and load the file contents into memory
            self.dataobs.load(df.uri)

        if self.cd_file is not None:
            # finally, the data covariance
            try:
                # get the path to the file
                cf = ifs[self.cd_file]
            # if the file doesn't exist
            except ifs.NotFoundError:
                # grab my error channel
                channel = self.error
                # complain
                channel.log(f"missing data covariance matrix: no '{self.cd_file}'")
                # and raise the exception again
                raise
            # if all goes well
            else:
                # allocate the matrix
                self.cd = altar.matrix(shape=(self.observations, self.observations))
                # and load the file contents into memory
                self.cd.load(cf.uri)
        else:
            # use a constant covariance
            self.cd = self.cd_std
        # all done
        return self


    def initializeCovariance(self):
        """
        """
        from math import log, pi as π

        # copy dataobs from cpu to gpu
        self.gdataObs = altar.cuda.vector(source=self.dataobs, dtype=self.precision)
        # make a temp copy
        gDataVec = self.gdataObs.clone()
        
        # process cd info
        cd = self.cd
        observations = self.observations
        if isinstance(cd, altar.matrix):
            # copy cd to gpu
            self.gcd = altar.cuda.matrix(source=cd, dtype=self.precision)
            # inverse and Cholesky
            self.gcd_inv = self.gcd.clone()

            gCd_inv = self.gcd_inv
            # inverse
            gCd_inv.inverse()
            # Cholesky factorization/decomposition
            gCd_inv.Cholesky()
            # normalization
            logdet = libcuda.matrix_logdet_triangular(gCd_inv.data)
            self.normalization = -0.5*log(2*π)*observations + logdet

            # merge cd to data
            altar.cuda.cublas.trmv(A=gCd_inv, x=gDataVec)
            
        elif isinstance(cd, float):
            # cd is standard deviation 
            self.normalization = -0.5*log(2*π*cd)*observations;
            self.gcd_inv = 1.0/self.cd # a scaler
            gDataVec *= self.gcd_inv

        # make duplicates of data vector to a matrix
        self.gdataObsBatch = altar.cuda.matrix(shape=(self.samples, self.observations), dtype=self.precision)
        self.gdataObsBatch.duplicateVector(src=gDataVec)

        # all done 
        return self


    # local variables
    # from cpu
    # dataobs = None
    # cd = None
    # cd_inv = None
    precision = None
    gdataObs = None
    gdataObsBatch = None
    gcd = None
    gcd_inv = None

# end of file
