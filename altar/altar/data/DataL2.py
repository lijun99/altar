# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-2020 parasim inc
# (c) 2010-2020 california institute of technology
# all rights reserved
#
# Author(s): Lijun Zhu

# the package
import altar
# my protocol
from .DataObs import DataObs as data


# declaration
class DataL2(altar.component, family="altar.data.datal2", implements=data):
    """
    The observed data with L2 norm
    """

    data_file = altar.properties.path(default="data.txt")
    data_file.doc = "the name of the file with the observations"

    observations = altar.properties.int(default=1)
    observations.doc = "the number of observed data"

    cd_file = altar.properties.path(default=None)
    cd_file.doc = "the name of the file with the data covariance matrix"

    cd_std = altar.properties.float(default=1.0)
    cd_std.doc = "the constant covariance for data"

    merge_cd_with_data = altar.properties.bool(default=False)
    merge_cd_with_data.doc = "whether to merge cd with data"

    norm = altar.norms.norm()
    norm.default = altar.norms.l2()
    norm.doc = "the norm to use when computing the data log likelihood"

    @altar.export
    def initialize(self, application):
        """
        Initialize data obs from model
        """
        # get the input path from model

        self.error = application.error
        # get the number of samples
        self.samples = application.job.chains
        # load the data and covariance
        self.ifs = application.pfs["inputs"]
        self.loadData()
        # compute inverse of covariance, normalization
        self.initializeCovariance(cd=self.cd)
        # all done
        return self

    def evalLikelihood(self, prediction, likelihood, residual=True, batch=None):
        """
        compute the datalikelihood for prediction (samples x observations)
        """

        #depending on convenience, users can
        # copy dataobs to their model and use the residual as input of prediction
        # or compute prediction from forward model and subtract the dataobs here

        batch = batch if batch is not None else likelihood.shape

        # go through the residual of each sample
        for idx in range(batch):
            # extract it
            dp = prediction.getRow(idx)
            # subtract the dataobs if residual is not pre-calculated
            if not residual:
                dp -= self.dataobs
            if self.merge_cd_with_data:
                # cd already merged, no need to multiply it by cd
                likelihood[idx] = self.normalization - 0.5 * self.norm.eval(v=dp)
            else:
                likelihood[idx] = self.normalization - 0.5 * self.norm.eval(v=dp, sigma_inv=self.cd_inv)
        # all done
        return self


    def dataobsBatch(self):
        """
        Get a batch of duplicated dataobs
        """
        if self.dataobs_batch is None:
            self.dataobs_batch = altar.matrix(shape=(self.samples, self.observations))
        # for each sample
        for sample in range(self.samples):
            # make the corresponding column a copy of the data vector
            self.dataobs_batch.setColumn(sample, self.dataobs)
        return self.dataobs_batch


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
        return


    def initializeCovariance(self, cd):
        """
        For a given data covariance cd, compute L2 likelihood normalization, inverse of cd in Cholesky decomposed form,
        and merge cd with data observation, d-> L*d with cd^{-1} = L L*
        :param cd:
        :return:
        """
        # grab the number of observations
        observations = self.observations

        if isinstance(cd, altar.matrix):
            # normalization
            self.normalization = self.computeNormalization(observations=observations, cd=cd)
            # inverse matrix
            self.cd_inv = self.computeCovarianceInverse(cd=cd)
            # merge cd to data
            if self.merge_cd_with_data:
                Cd_inv = self.cd_inv
                self.dataobs = altar.blas.dtrmv( Cd_inv.upperTriangular, Cd_inv.opNoTrans, Cd_inv.nonUnitDiagonal,
                    Cd_inv, self.dataobs)

        elif isinstance(cd, float):
            # cd is standard deviation
            from math import log, pi as π
            self.normalization = -0.5*log(2*π*cd)*observations;
            self.cd_inv = 1.0/self.cd
            if self.merge_cd_with_data:
                self.dataobs *= self.cd_inv

        # all done
        return self

    def updateCovariance(self, cp):
        """
        Update data covariance with cp, cd -> cd + cp
        :param cp: a matrix with shape (obs, obs)
        :return:
        """
        # make a copy of cp
        cchi = cp.clone()
        # add cd (scalar or matrix)
        cchi += self.cd
        self.initializeCovariance(cd=cchi)
        return self

    def computeNormalization(self, observations, cd):
        """
        Compute the normalization of the L2 norm
        """
        # support
        from math import log, pi as π
        # make a copy of cd
        cd = cd.clone()
        # compute its LU decomposition
        decomposition = altar.lapack.LU_decomposition(cd)
        # use it to compute the log of its determinant
        logdet = altar.lapack.LU_lndet(*decomposition)

        # all done
        return - (log(2*π)*observations + logdet) / 2;


    def computeCovarianceInverse(self, cd):
        """
        Compute the inverse of the data covariance matrix
        """
        # make a copy so we don't destroy the original
        cd = cd.clone()
        # perform the LU decomposition
        lu = altar.lapack.LU_decomposition(cd)
        # invert; this creates a new matrix
        inv = altar.lapack.LU_invert(*lu)
        # compute the Cholesky decomposition
        inv = altar.lapack.cholesky_decomposition(inv)

        # and return it
        return inv

    # local variables
    normalization = 0
    ifs = None
    samples = None
    dataobs = None
    dataobs_batch = None
    cd = None
    cd_inv = None
    error = None

# end of file
