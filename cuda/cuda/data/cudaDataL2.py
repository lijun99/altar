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
import numpy as np
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

    dtype_cd = altar.properties.str(default=None)
    dtype_cd.doc = "the data type (float32/64) for Cd computations if different from others"

    datafile_dataset = altar.properties.str(default=None)
    datafile_dataset.doc = "the datafile dataset name if the file type is h5"

    mask_dataset = altar.properties.str(default=None)
    mask_dataset.doc = "the mask dataset name if the file type is h5"

    # the norm to use for computing the data log likelihood
    # the only implementation that works for now
    norm = altar.cuda.norms.norm()
    norm.default = altar.cuda.norms.l2()
    norm.doc = "l2 norm for calculating likelihood"

    # constant variables
    # whether merge cd/cp to observed data; to avoid repeating cd^{-1} d,
    #     e.g. for models with d = G m
    merge_cd_to_data = True
    # whether to allocate a data matrix (samples, observations) to keep copies of the data vector
    #     only set it to false when the model deals with large amount of data
    provide_batched_data = True

    @altar.export
    def initialize(self, application):
        """
        Initialize data obs from model
        """
        # load the gpu part at first
        # initCovariance depends on precision
        self.device = application.controller.worker.device
        self.precision = application.job.gpuprecision

        self.dtype_cd = self.dtype_cd or self.precision

        # get the input path from model
        self.ifs = application.pfs["inputs"]
        self.error = application.error
        # get the number of samples
        self.samples = application.job.chains

        # get the number of observations
        observations = self.observations

        # load the observed data to numpy array
        self.dataobs = self.loadFile(filename=self.data_file, shape=observations,
                                     dataset=self.datafile_dataset)

        # load a mask inside the .h5 file
        if self.mask_dataset is None:
            self.mask = None
            if np.isfinite(self.dataobs).sum() != self.dataobs.size:
                raise ValueError("Found non-finite values in dataobs but no mask provided!")
        else:
            self.mask = self.loadFile(filename=self.data_file, shape=observations,
                                      dataset=self.mask_dataset, dtype=bool)
            if np.isfinite(self.dataobs[self.mask]).sum() != self.mask.sum():
                raise ValueError(
                    "Found non-finite values in dataobs even after applying the mask!")
            self.dataobs[~self.mask] = 0

        # load the data covariance
        if self.cd_file is not None:
            self.cd = self.loadFile(filename=self.cd_file, shape=(observations, observations))
        else:
            # use a constant covariance
            self.gcd_inv = 1.0/self.cd_std
            # in model, use the following to check whether cd is a constant or matrix
            #     cd_inv = self.dataobs.gcd_inv
            #     if isinstance(cd_inv, float):

            # # delay creation of cd only if cp is considered
            # self.cd = numpy.zeros(shape=(observations, observations), dtype=self.dtype_cd)
            # numpy.fill_diagonal(self.cd, self.cd_std**2)

        # compute inverse of covariance, normalization
        self.initializeCovariance()

        # all done
        return self

    def cuEvalLikelihood(self, prediction, likelihood, residual=True, batch=None, weight=None):
        """
        compute the datalikelihood for prediction
        :param prediction: (samples x observations) input of predicted data
        :param likelihood: (samples) pre-allocated likelihood/norm
        :param residual: whether prediction is already subtracted by observed data
        :param batch: number of (first few) samples to be computed
        :return: likelihood
        """

        # get the batch / number of samples
        batch = batch or prediction.shape[0]

        # depending on convenience, users may
        # either copy dataobs to their model and use the residual as input of prediction
        # or compute prediction from forward model and subtract the dataobs here

        # subtract dataobs from prediction to get residual
        if not residual:
            # check whether batched data is available
            if self.provide_batched_data:
                prediction -= self.gdataObsBatch
            else:
                # not, create it on the fly
                gdataObsBatch = altar.cuda.matrix(
                    shape=(self.samples, self.observations), dtype=self.precision)
                gdataObsBatch.duplicateVector(src=self.self.gDataVec)
                prediction -= gdataObsBatch

        # call L2 norm to calculate the likelihood
        normalization = self.normalization  # norm constant

        likelihood = self.norm.cuEvalLikelihood(
            data=prediction, constant=normalization, out=likelihood, batch=batch, weight=weight)

        # all done
        return likelihood

    @property
    def cd_inv(self):
        """
        Inverse of data covariance, in Cholesky decomposed form
        """
        return self.gcd_inv

    def release_cd(self):
        """
        release gcd_inv
        """
        self.gcd_inv = None
        return

    @property
    def dataobsBatch(self):
        """
        A batch of duplicated observations
        """
        return self.gdataObsBatch

    def loadFile(self, filename, shape, dataset=None, dtype=None):
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
                cpuData = numpy.loadtxt(file.uri.path, dtype=dtype).reshape(shape)
            # binary data
            elif suffix == '.bin' or suffix == '.dat':
                # read and reshape, users need to check the precision
                cpuData = numpy.fromfile(file.uri.path, dtype=dtype).reshape(shape)
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
                cpuData = numpy.asarray(h5file.get(dataset), dtype=dtype).reshape(shape)
                h5file.close()
        # all done
        return cpuData

    def initializeCovariance(self):
        """
        initialize gpu data and data covariance
        """
        # initialize arrays kept in GPU memory
        observations = self.observations
        samples = self.samples

        self.gDataVec = altar.cuda.vector(shape=observations, dtype=self.precision)

        if self.provide_batched_data:
            self.gdataObsBatch = altar.cuda.matrix(
                shape=(samples, observations), dtype=self.precision)

        if self.mask is None:
            self.gWeight = None
        else:
            self.gWeight = altar.cuda.vector(source=self.mask.astype(self.precision))

        self.updateCovariance()

        # all done
        return self

    def updateCovariance(self, cp=None):

        # initialize Cd
        if isinstance(self.gcd_inv, float):
            self.updateCovarianceConstantCd()
        else:
            self.updateCovarianceMat(cp=cp)

        return self

    def updateCovarianceConstantCd(self):
        """
        initialize covariance related parameters for a constant cd w cd_std
        :return:
        """

        from math import log, pi as π
        # process cd info
        cdinv = self.gcd_inv
        observations = self.observations
        # L2 normalization
        self.normalization = (-0.5 * log(2 * π) + log(cdinv)) * observations

        # prepare self.gdataObsBatch
        # load data to gpu
        self.gDataVec.copy_from_host(source=self.dataobs)

        # merge Cchi to data
        if self.merge_cd_to_data:
            self.gDataVec = self.mergeCdtoData(cd_inv=self.gcd_inv, data=self.gDataVec)

        # make duplicates of data vector to a matrix
        if self.provide_batched_data:
            self.gdataObsBatch.duplicateVector(src=self.gDataVec)

        # all done
        return self

    def updateCovarianceMat(self, cp=None):
        """
        Update the data covariance C_chi = Cd + Cp
        :param cp: cuda matrix with shape(obs, obs), data covariance due to model uncertainty
        :return:
        """

        from math import log, pi as π
        # process cd info
        observations = self.observations

        # obtain Cd from cpu
        gCchi = altar.cuda.matrix(source=self.cd, dtype=self.dtype_cd)

        # add Cp if provided
        if cp is not None:
            # check cp data type
            gCp = cp if cp.dtype == self.dtype_cd else cp.copy_to_device(dtype=self.dtype_cd)
            # add cp to cd
            gCchi += gCp

        # self.checkPositiveDefiniteness(matrix=gCchi, name='Cchi')

        # Inverse
        gCchi.inverse()

        # self.checkPositiveDefiniteness(matrix=gCchi, name='Cchi inverse')

        # Choleseky decomposition
        gCchi.Cholesky(uplo=cublas.FillModeUpper)

        # normalization
        logdet = libcuda.matrix_logdet_triangular(gCchi.data)
        self.normalization = -0.5*log(2*π)*observations + logdet

        # keep a copy of inverse Cchi to gcd_inv for other operations
        self.gcd_inv = (gCchi if gCchi.dtype == self.precision
                        else gCchi.copy_to_host(dtype=self.precision))

        # load data to gpu
        self.gDataVec = altar.cuda.vector(source=self.dataobs, dtype=self.precision)

        # merge Cchi to data
        self.gDataVec = self.mergeCdtoData(cd_inv=self.gcd_inv, data=self.gDataVec)

        # make duplicates of data vector to a matrix
        self.gdataObsBatch.duplicateVector(src=self.gDataVec)

        # all done
        return self

    def checkPositiveDefiniteness(self, matrix, name=None):
        """
        Check positive definiteness of a GPU matrix
        :param matrix: a real symmetric (GPU) matrix
        :return: true or false
        """
        import numpy
        name = name or 'Matrix'
        cm = matrix.copy_to_host(type='numpy')
        eval = numpy.linalg.eigvalsh(cm)
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
        :param cd_inv: the inverse of covariance matrix in Cholesky-decomposed form,
                       with lower matrix filled
        :param data: raw observed data
        :return:  cd_inv*data, a cuda vector
        """

        # make a copy of observed data
        self.gDataVec = data.clone()

        if isinstance(cd_inv, float):
            self.gDataVec *= cd_inv
        else:
            # Cd^{-1} = LL^T
            # d -> d (1, obs) x L (obs, obs)
            cublas.trmv(A=cd_inv,
                        x=self.gDataVec,
                        uplo=cublas.FillModeUpper,
                        transa=cublas.OpTrans
                        )
            # all done
        return self.gDataVec

    # local variables
    # # from cpu class
    # dataobs = None
    # cd = None
    # cd_inv = None
    # gpu
    normalization = 0
    precision = None
    gDataVec = None  # keep the data vector in memory
    gdataObsBatch = None  # keep the duplicates (samples) of data vectors in memory
    gcd_inv = None  # kept in memory, can be released upon request

# end of file
