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
from altar.cuda import cublas as cublas

import numpy

# my protocol
from altar.data.DataL2 import DataL2



# declaration
class cudaDataL2(DataL2, family="altar.data.cudadatal2"):
    """
    The observed data with L2 norm
    """

    # configuration states copied from cpu counterpart
    data_file = altar.properties.path(default="data.txt")
    data_file.doc = "the name of the file with the observations"

    observations = altar.properties.int(default=1)
    observations.doc = "the number of observed data"

    cd_file = altar.properties.path(default=None)
    cd_file.doc = "the name of the file with the data covariance matrix"

    cd_std = altar.properties.float(default=1.0)
    cd_std.doc = "the constant covariance for data, sigma^2"

    merge_cd_to_data = altar.properties.bool(default=True)
    merge_cd_to_data.doc = "whether to merge Cd with observed data"


    # the norm to use for computing the data log likelihood
    # the only implementation that works for now
    norm = altar.cuda.norms.norm()
    norm.default = altar.cuda.norms.l2()
    norm.doc = "l2 norm for calculating likelihood"

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

        # load the observed data
        self.dataobs = self.loadFile(filename=self.data_file, shape=self.observations)

        # load the data covariance
        if self.cd_file is not None:
            self.cd = self.loadFile(filename=self.cd_file, shape=(self.observations, self.observations))
        else:
            # use a constant covariance
            self.cd = self.cd_std

        # compute inverse of covariance, normalization
        self.initializeCovariance()

        # all done
        return self


    def cuEvalLikelihood(self, prediction, likelihood, residual=True, batch=None):
        """
        compute the datalikelihood for prediction (samples x observations)
        """

        # get the batch / number of samples
        batch = batch or prediction.shape[0]

        # depending on convenience, users can
        # either copy dataobs to their model and use the residual as input of prediction
        #
        # or compute prediction from forward model and subtract the dataobs here

        # subtract dataobs from prediction to get residual
        if not residual:
            prediction -= self.gdataObsBatch

        # call L2 norm to calculate the likelihood
        likelihood = self.norm.cuEvalLikelihood(data=prediction, constant=self.normalization,
            out=likelihood, batch=batch)

        # all done
        return likelihood

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

    def loadFile(self, filename, shape, dataset=None):
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
                cpuData = numpy.loadtxt(file.uri.path, dtype = self.precision).reshape(shape)
            # binary data
            elif suffix == '.bin' or suffix == '.dat':
                # read and reshape, users need to check the precision
                cpuData = numpy.fromfile(file.uri.path, dtype=self.precision).reshape(shape)
            # hdf5 file
            elif suffix == '.h5':
                # get support
                import h5py
                # open
                h5file = h5py.File(file.uri.path, 'r')
                # get the desired dataset
                if dataset is None:
                    # if not provided, assume the first dataset available
                    dataset = list(h5file.keys())[0]
                cpuData = numpy.asarray(h5file.get(dataset), dtype=self.precision).reshape(shape)
                h5file.close()
        # all done
        return cpuData


    def initializeCovariance(self):
        """
        initialize gpu data and data covariance
        """

        # copy dataobs from cpu to gpu
        self.gdataObs = altar.cuda.vector(source=self.dataobs, dtype=self.precision)
        # allocate an array of duplicated dataobs
        self.gdataObsBatch = altar.cuda.matrix(shape=(self.samples, self.observations), dtype=self.precision)

        # process cd info
        cd = self.cd
        observations = self.observations
        if isinstance(cd, float):
            # cd is standard deviation/scalar
            cd_mat = numpy.zeros(shape=(observations, observations), dtype=self.precision)
            numpy.fill_diagonal(cd_mat, cd)
            self.gcd = altar.cuda.matrix(source=cd_mat, dtype=self.precision)
        # cd is a matrix
        elif isinstance(cd, numpy.ndarray):
            # copy cd to gpu
            self.gcd = altar.cuda.matrix(source=cd, dtype=self.precision)

        self.gcd_inv = altar.cuda.matrix(shape=self.gcd.shape, dtype=self.precision)

        # initialize with Cd only
        self.updateCovariance()

        # all done
        return self

    def updateCovariance(self, cp=None):
        """
        Update the data covariance C_chi = Cd + Cp
        :param cp: cuda matrix with shape(obs, obs), data covariance due to model uncertainty
        :return:
        """

        from math import log, pi as π

        # get references
        observations = self.observations
        # get the numerical precision
        dtype = self.gcd_inv.dtype

        # prepare Cchi, Cp
        if dtype == 'float64':
            Cchi = self.gcd_inv
            Cp = cp
        else:
            Cchi = altar.cuda.matrix(shape=self.gcd_inv.shape, dtype='float64')
            Cp = cp.copy_to_device(dtype='float64') if cp is not None else None

        # copy cd over, convert dtype if neccesary
        Cchi = self.gcd.copy_to_device(out=Cchi)
        # self.checkPostivieDefiniteness(matrix=Cchi, name='Cd')
        # add Cp
        if Cp is not None:
            Cchi += Cp
        #   self.checkPostivieDefiniteness(matrix=Cp, name='Cp')
        # Inverse and Choleseky decomposition
        # self.checkPostivieDefiniteness(matrix=Cchi, name='Cchi')
        Cchi.inverse()
        # self.checkPostivieDefiniteness(matrix=Cchi, name='Cchi inverse')
        Cchi.Cholesky(uplo=cublas.FillModeUpper)

        if dtype != 'float64':
            # copy back to single precision version
            self.gcd_inv = Cchi.copy_to_device(out=self.gcd_inv)

        # use the new reference
        Cchi = self.gcd_inv
        # normalization
        logdet = libcuda.matrix_logdet_triangular(Cchi.data)
        self.normalization = -0.5*log(2*π)*observations + logdet

        # merge Cchi to data
        if self.merge_cd_to_data:
            gDataVec = self.mergeCdtoData(cd_inv=Cchi, data=self.gdataObs)
        else:
            gDataVec = self.gdataObs

        # make duplicates of data vector to a matrix
        self.gdataObsBatch.duplicateVector(src=gDataVec)

        # all done
        return self

    def checkPostivieDefiniteness(self, matrix, name=None):
        """
        Check positive definiteness of a GPU matrix
        :param matrix: a real symmetric (GPU) matrix
        :return: true or false
        """
        import numpy
        name = name or 'Matrix'
        cm = matrix.copy_to_host(type='numpy')
        eval= numpy.linalg.eigvalsh(cm)
        n = eval.shape[0]
        if eval[n-1] < 0:
            print(name, " is not positive definite!")
            print(eval)
            return False
        print(name, eval.min(), eval.max())
        return True

    def mergeCdtoData(self, cd_inv, data):
        """
        Merge the data covariance matrix to observed data
        :param cd_inv: the inverse of covariance matrix in Cholesky-decomposed form, with Lower matrix filled
        :param data: raw observed data
        :return:  cd_inv*data, a cuda vector
        """

        # make a copy of observed data
        gDataVec = data.clone()

        # cd is a constant
        if isinstance(cd_inv, float):
            gDataVec *= cd_inv
        elif isinstance(cd_inv, altar.cuda.matrix):
            # Cd^{-1} = LL^T
            # d -> d (1, obs) x L (obs, obs)
            cublas.trmv(A=cd_inv, x=gDataVec,
                        uplo=cublas.FillModeUpper,
                        transa = cublas.OpTrans
                        )
        # all done
        return gDataVec

    # local variables
    # from cpu
    # dataobs = None
    # cd = None
    # cd_inv = None
    normalization = 0
    precision = None
    gdataObs = None
    gdataObsBatch = None
    gcd = None
    gcd_inv = None

# end of file
