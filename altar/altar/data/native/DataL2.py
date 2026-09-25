# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-2025 parasim inc
# (c) 2010-2025 california institute of technology
# all rights reserved
#
# Author(s): Lijun Zhu

# the package
import altar


# the declaration
class DataL2:
    """
    The cpu implementation of observed data with L2 norm

    My configuration (data_file, observations, cd_file, cd_std, merge_cd_with_data, norm) is
    copied down from the shim, once, before {initialize} runs.
    """

    @altar.export
    def initialize(self, application):
        """
        Initialize data obs from model
        """
        # get the input path from model
        self.error = application.error
        self.info = application.info
        # get the number of samples
        self.samples = application.job.chains
        # load the data and covariance
        self.ifs = application.pfs["inputs"]
        # set up my file reader/writer
        self.io = altar.io.FileIO(ifs=self.ifs, error=self.error)
        self.load_data()
        # compute inverse of covariance, normalization
        self.initialize_covariance(cd=self.cd)
        # all done
        return self


    def eval_likelihood(self, prediction, likelihood, residual=True, batch=None):
        """
        compute the datalikelihood for prediction (samples x observations)
        """
        # depending on convenience, users can
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
                norm = self.norm.eval(v=dp)
            else:
                norm = self.norm.eval(v=dp, sigma_inv=self.cd_inv)
            # {norm.eval} returns the (unsquared) L2 norm; the Gaussian log-likelihood needs
            # its square
            likelihood[idx] = self.normalization - 0.5 * norm * norm
        # all done
        return self


    @property
    def dataobs_batch(self):
        """
        Get a batch of duplicated dataobs

        The original cpu implementation shadowed this method with a same-named class
        attribute default (`dataobs_batch = None`), making it permanently unreachable; this
        is a property instead, matching the cuda side's own {dataobs_batch} property.
        """
        if self._dataobs_batch is None:
            self._dataobs_batch = altar.matrix(shape=(self.samples, self.observations))
        # for each sample
        for sample in range(self.samples):
            # make the corresponding column a copy of the data vector
            self._dataobs_batch.setColumn(sample, self.dataobs)
        return self._dataobs_batch


    def load_data(self):
        """
        load data and covariance
        """
        # next, the observations
        self.dataobs = self.io.load(filename=self.data_file, shape=self.observations)

        if self.cd_file is not None:
            # finally, the data covariance
            self.cd = self.io.load(
                filename=self.cd_file,
                shape=(self.observations, self.observations),
            )
        else:
            # use a constant covariance
            self.cd = self.cd_std
        return


    def initialize_covariance(self, cd):
        """
        For a given data covariance cd, compute L2 likelihood normalization, inverse of cd
        in Cholesky decomposed form, and merge cd with data observation, d-> L*d with
        cd^{-1} = L L*
        """
        # grab the number of observations
        observations = self.observations

        if isinstance(cd, altar.matrix):
            # normalization
            self.normalization = self.compute_normalization(observations=observations, cd=cd)
            # inverse matrix
            self.cd_inv = self.compute_covariance_inverse(cd=cd)
            # merge cd to data
            if self.merge_cd_with_data:
                Cd_inv = self.cd_inv
                self.dataobs = altar.blas.dtrmv(
                    Cd_inv.upperTriangular, Cd_inv.opNoTrans, Cd_inv.nonUnitDiagonal,
                    Cd_inv, self.dataobs)

        elif isinstance(cd, float):
            # cd is standard deviation
            from math import log, pi as π
            self.normalization = -0.5 * log(2 * π * cd) * observations
            self.cd_inv = 1.0 / self.cd
            if self.merge_cd_with_data:
                self.dataobs *= self.cd_inv

        # all done
        return self


    def update_covariance(self, cp=None):
        """
        Update data covariance with cp, cd -> cd + cp
        """
        # make a copy of cp
        cchi = cp.clone()
        # add cd (scalar or matrix)
        cchi += self.cd
        self.initialize_covariance(cd=cchi)
        return self


    def compute_normalization(self, observations, cd):
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
        return -(log(2 * π) * observations + logdet) / 2


    def compute_covariance_inverse(self, cd):
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


    # configuration, copied down from the shim by {_makeImpl}
    data_file = None
    observations = None
    cd_file = None
    cd_std = None
    merge_cd_with_data = None
    norm = None

    # local variables
    normalization = 0
    ifs = None
    io = None  # my file reader/writer
    samples = None
    dataobs = None
    _dataobs_batch = None
    cd = None
    cd_inv = None
    error = None
    info = None


# end of file
