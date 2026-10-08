# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# the package
import numpy
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


    def eval_likelihood(self, prediction, likelihood, residual=True, batch=None, whitened=True):
        """
        compute the datalikelihood for prediction (samples x observations); {whitened=False}
        marks {prediction} as raw even when {merge_cd_with_data} is set
        """
        # depending on convenience, users can
        # copy dataobs to their model and use the residual as input of prediction
        # or compute prediction from forward model and subtract the dataobs here
        batch = batch if batch is not None else likelihood.shape
        # whether {prediction} has cd merged into it
        merged = self.merge_cd_with_data and whitened
        # the data to compare it against
        data = self.dataobs if merged or not self.merge_cd_with_data else self._observed_gsl

        # the residuals of all the samples at once, (batch x observations)
        dp = numpy.asarray(prediction)[:batch]
        # subtract the dataobs if residual is not pre-calculated
        if not residual:
            dp = dp - numpy.asarray(data)
        # cd already merged, no need to multiply it by cd
        sigma_inv = None if merged else self.cd_inv
        self.norm.eval_likelihood(
            v=dp, constant=self.normalization, sigma_inv=sigma_inv, batch=batch, out=likelihood,
            weight=self.mask)
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
        self.dataobs = self.io.load(
            filename=self.data_file, shape=self.observations, dataset=self.datafile_dataset)
        # a raw copy, kept since {initialize_covariance} may merge the covariance into {dataobs}
        self._observed = numpy.array(self.dataobs, dtype=float)
        # the valid observations, if some are masked
        self.load_mask()
        # the raw copy, as a gsl vector, for raw predictions against merged data
        self._observed_gsl = self.io.toGsl(self._observed.copy())

        if self.cd_file is not None:
            if self.mask is not None:
                self.error.log("a data mask only works with a constant covariance, cd_std")
                raise SystemExit(1)
            # finally, the data covariance
            self.cd = self.io.load(
                filename=self.cd_file,
                shape=(self.observations, self.observations),
            )
        else:
            # use a constant covariance
            self.cd = self.cd_std
        return


    def load_mask(self):
        """
        Load the mask of valid observations from {mask_dataset}, and zero the masked data
        """
        # no mask: all the data must be valid
        if self.mask_dataset is None:
            self.mask = None
            if not numpy.isfinite(self._observed).all():
                self.error.log(f"non-finite values in '{self.data_file}', but no mask_dataset")
                raise SystemExit(1)
            return self

        mask = self.io.load(
            filename=self.data_file, shape=self.observations, dataset=self.mask_dataset)
        self.mask = numpy.asarray(mask) != 0
        if not numpy.isfinite(self._observed[self.mask]).all():
            self.error.log(f"non-finite values in '{self.data_file}' that are not masked")
            raise SystemExit(1)
        # masked values are left out of the likelihood; zero them so they stay finite
        self._observed[~self.mask] = 0
        self.dataobs = self.io.toGsl(self._observed.copy())
        # all done
        return self


    def observed(self):
        """
        The raw observed data, as a numpy vector
        """
        return self._observed


    def sigma(self):
        """
        The standard deviation of each observation, as a numpy vector
        """
        if isinstance(self.cd, float):
            return numpy.full(self.observations, self.cd)
        return numpy.sqrt(numpy.diag(numpy.asarray(self.cd)))


    def sigma_chi(self):
        """
        The standard deviation of each observation under C_chi, as a numpy vector
        """
        return self.sigma() if self._chi_variance is None else numpy.sqrt(self._chi_variance)


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
                self.dataobs = self.io.toGsl(self._observed.copy())
                self.dataobs = altar.blas.dtrmv(
                    Cd_inv.upperTriangular, Cd_inv.opNoTrans, Cd_inv.nonUnitDiagonal,
                    Cd_inv, self.dataobs)

        elif isinstance(cd, float):
            # cd is standard deviation
            from math import log, pi as π
            # only the valid observations count
            if self.mask is not None:
                observations = int(self.mask.sum())
            self.normalization = -0.5 * log(2 * π) * observations - observations * log(cd)
            self.cd_inv = 1.0 / cd
            if self.merge_cd_with_data:
                self.dataobs = self.io.toGsl(self._observed * self.cd_inv)

        # all done
        return self


    def update_covariance(self, cp=None):
        """
        Use C_chi = C_d + {cp}, a numpy (observations x observations) array, from now on; back
        to C_d alone if {cp} is None
        """
        if cp is None:
            self._chi_variance = None
            return self.initialize_covariance(cd=self.cd)
        # a full C_chi mixes the masked observations into the valid ones
        if self.mask is not None:
            self.error.log("a data mask only works with a constant covariance, without a C_p")
            raise SystemExit(1)
        cd = self.cd
        if isinstance(cd, float):
            cchi = numpy.diag(numpy.full(self.observations, cd * cd))
        else:
            cchi = numpy.array(cd, dtype=float)
        cchi += numpy.asarray(cp, dtype=float)
        self._chi_variance = numpy.diag(cchi).copy()
        return self.initialize_covariance(cd=self.io.toGsl(cchi))


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
    datafile_dataset = None
    mask_dataset = None

    # local variables
    normalization = 0
    ifs = None
    io = None  # my file reader/writer
    samples = None
    dataobs = None
    _dataobs_batch = None
    cd = None
    cd_inv = None
    _chi_variance = None # diag(C_chi), when a C_p is part of it
    _observed_gsl = None # the raw observed data, as a gsl vector
    mask = None # the valid observations, a numpy bool vector; None if all are valid
    error = None
    info = None


# end of file
