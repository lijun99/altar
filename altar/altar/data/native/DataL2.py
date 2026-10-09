# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# externals
from __future__ import annotations
import math
import typing
import numpy
# the package
import altar

if typing.TYPE_CHECKING:
    import journal
    import pyre
    from altar.norms.L2 import L2
    from altar.shells.Application import Application


# the declaration
class DataL2:
    """
    The cpu implementation of observed data with L2 norm

    My configuration (data_file, observations, cd_file, cd_std, merge_cd_with_data, norm) is
    copied down from the shim, once, before {initialize} runs.
    """

    @altar.export
    def initialize(self, application: Application) -> typing.Self:
        """
        Initialize data obs from model
        """
        # get the input path from model
        self.error = application.error
        self.info = application.info
        # get the number of samples
        self.samples = application.job.chains
        # and the precision of the residuals; the densities are always in double precision
        self.precision = application.job.precision
        # load the data and covariance
        self.ifs = application.pfs["inputs"]
        # set up my file reader/writer
        self.io = altar.io.FileIO(ifs=self.ifs, error=self.error)
        self.load_data()
        # compute inverse of covariance, normalization
        self.initialize_covariance(cd=self.cd)
        # all done
        return self


    def eval_likelihood(self, prediction: numpy.ndarray, likelihood: numpy.ndarray,
                        residual: bool = True, batch: int | None = None,
                        whitened: bool = True) -> typing.Self:
        """
        Fill the first {batch} entries of {likelihood} with the data log likelihoods of the
        (samples x observations) {prediction}, the residuals if {residual}; {whitened=False}
        marks {prediction} as raw even when {merge_cd_with_data} is set
        """
        batch = likelihood.shape[0] if batch is None else batch
        # whether {prediction} has cd merged into it
        merged = self.merge_cd_with_data and whitened
        # the data to compare it against
        data = self.dataobs if merged or not self.merge_cd_with_data else self._raw
        # the residuals of all the samples at once, (batch x observations)
        dp = prediction[:batch]
        # subtract the dataobs if residual is not pre-calculated
        if not residual:
            dp = dp - data
        # cd already merged, no need to multiply it by cd
        sigma_inv = None if merged else self.cd_inv
        self.norm.eval_likelihood(
            v=dp, constant=self.normalization, sigma_inv=sigma_inv, batch=batch, out=likelihood,
            weight=self.mask)
        # all done
        return self


    @property
    def dataobs_batch(self) -> numpy.ndarray:
        """
        The (samples x observations) array with a copy of {dataobs} in each row
        """
        return numpy.tile(self.dataobs, (self.samples, 1))


    def load_data(self) -> None:
        """
        load data and covariance
        """
        # the observations; {_observed} keeps them raw, since {initialize_covariance} may merge
        # the covariance into {dataobs}
        self._observed = self.io.load(
            filename=self.data_file, shape=self.observations, dataset=self.datafile_dataset)
        # the valid observations, if some are masked
        self.load_mask()
        # the raw data, in my precision, for the residuals
        self._raw = self._observed.astype(self.precision)
        self.dataobs = self._raw.copy()

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


    def load_mask(self) -> typing.Self:
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
        self.mask = mask != 0
        if not numpy.isfinite(self._observed[self.mask]).all():
            self.error.log(f"non-finite values in '{self.data_file}' that are not masked")
            raise SystemExit(1)
        # masked values are left out of the likelihood; zero them so they stay finite
        self._observed[~self.mask] = 0
        # all done
        return self


    def observed(self) -> numpy.ndarray:
        """
        The raw observed data
        """
        return self._observed


    def sigma(self) -> numpy.ndarray:
        """
        The standard deviation of each observation
        """
        if isinstance(self.cd, float):
            return numpy.full(self.observations, self.cd)
        return numpy.sqrt(numpy.diag(self.cd))


    def sigma_chi(self) -> numpy.ndarray:
        """
        The standard deviation of each observation under C_chi
        """
        return self.sigma() if self._chi_variance is None else numpy.sqrt(self._chi_variance)


    def covariance(self) -> float | numpy.ndarray:
        """
        The covariance in effect, C_d or C_chi = C_d + C_p: an (observations x observations)
        array, or a float, the common variance, when it is a constant times the identity
        """
        if self._covariance is not None:
            return self._covariance
        if isinstance(self.cd, float):
            return self.cd * self.cd
        return numpy.array(self.cd, dtype=float)


    def initialize_covariance(self, cd: float | numpy.ndarray) -> typing.Self:
        """
        For a given data covariance {cd}, a standard deviation or a full matrix, compute the
        normalization of the L2 likelihood and L, the lower Cholesky factor of cd^{-1} = L L^T,
        and merge it into the data, d -> L^T d, if asked to
        """
        # grab the number of observations
        observations = self.observations

        if isinstance(cd, numpy.ndarray):
            # normalization
            self.normalization = self.compute_normalization(observations=observations, cd=cd)
            # the factor of the inverse, computed in double precision, used in mine
            L = self.compute_covariance_inverse(cd=cd)
            self.cd_inv = L.astype(self.precision)
            # merge it into the data
            if self.merge_cd_with_data:
                self.dataobs = (L.T @ self._observed).astype(self.precision)
        else:
            # cd is standard deviation
            # only the valid observations count
            if self.mask is not None:
                observations = int(self.mask.sum())
            self.normalization = -0.5 * math.log(2 * math.pi) * observations - observations * math.log(cd)
            self.cd_inv = 1.0 / cd
            if self.merge_cd_with_data:
                self.dataobs = (self._observed * self.cd_inv).astype(self.precision)

        # all done
        return self


    def update_covariance(self, cp: numpy.ndarray | None = None) -> typing.Self:
        """
        Use C_chi = C_d + {cp}, an (observations x observations) array, from now on; back to
        C_d alone if {cp} is None
        """
        if cp is None:
            self._chi_variance = None
            self._covariance = None
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
        self._covariance = cchi
        return self.initialize_covariance(cd=cchi)


    def compute_normalization(self, observations: int, cd: numpy.ndarray) -> float:
        """
        Compute the normalization of the L2 norm
        """
        sign, logdet = numpy.linalg.slogdet(cd)
        return -(math.log(2 * math.pi) * observations + logdet) / 2


    def compute_covariance_inverse(self, cd: numpy.ndarray) -> numpy.ndarray:
        """
        L, the lower Cholesky factor of the inverse of the data covariance, cd^{-1} = L L^T
        """
        return numpy.linalg.cholesky(numpy.linalg.inv(cd))


    # configuration, copied down from the shim by {_makeImpl}
    data_file: str
    observations: int
    cd_file: str | None = None
    cd_std: float
    merge_cd_with_data: bool = False
    norm: L2
    datafile_dataset: str | None = None
    mask_dataset: str | None = None

    # local variables
    normalization: float = 0
    ifs: pyre.filesystem.Filesystem.Filesystem
    io: altar.io.FileIO  # my file reader/writer
    samples: int
    dataobs: numpy.ndarray  # the observed data, with the covariance merged in if asked to
    cd: float | numpy.ndarray
    cd_inv: float | numpy.ndarray  # 1/sigma, or L with cd^{-1} = L L^T
    _chi_variance: numpy.ndarray | None = None # diag(C_chi), when a C_p is part of it
    _covariance: numpy.ndarray | None = None # C_chi, when a C_p is part of it
    _observed: numpy.ndarray  # the raw observed data, in double precision
    _raw: numpy.ndarray  # the raw observed data, in my precision
    precision: str = "float64"  # the precision of the residuals
    mask: numpy.ndarray | None = None # the valid observations; None if all are valid
    error: journal.error
    info: journal.info


# end of file
