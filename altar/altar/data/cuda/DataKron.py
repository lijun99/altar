# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# externals
import math
import numpy
# the package
import altar
# my base class
from .DataL2 import DataL2
from ..native.DataKron import load_kronecker


class DataKron(DataL2):
    """
    The cuda implementation of observed data with a {Kronecker} covariance: the data merged with
    it, always, as {DataL2} on cuda; {cd_inv} is its factor L, Cd^-1 = L L^T, applied on the
    host, as x @ L, by the models to their green's functions once, and here to raw residuals
    """

    @altar.export
    def initialize(self, application):
        """
        The observations and the factors of their covariance
        """
        self.precision = application.job.gpuprecision
        self.cd_dtype = self.cd_dtype or self.precision
        self.ifs = application.pfs["inputs"]
        self.error = application.error
        self.info = application.info
        self.samples = application.job.chains
        self.cd = load_kronecker(self)
        self.dataobs = self.load_file(
            filename=self.data_file, shape=self.observations, dataset=self.datafile_dataset,
            dtype="float64")
        if not numpy.isfinite(self.dataobs).all():
            self.error.log(f"non-finite values in '{self.data_file}'")
            raise SystemExit(1)
        self.mask = None
        self.initialize_covariance()
        return self


    def update_covariance(self, cp=None):
        """
        The normalization, the factor L of Cd^-1, and the data whitened by it
        """
        if cp is not None:
            self.error.log("datakron: a C_p needs a dense covariance, altar.data.datal2")
            raise SystemExit(1)
        self._chi_variance = None
        self._covariance = None
        self.normalization = -0.5 * (self.observations * math.log(2 * math.pi) + self.cd.logdet)
        self.cd_inv = self.cd.factor
        self._dataobs_merged = altar.cuda.managed(shape=(self.observations,), cell=self.precision)
        numpy.asarray(self._dataobs_merged)[:] = self.cd.whiten(self.dataobs)
        self._dataobs_batch = None
        return self


    def sigma(self):
        """
        The standard deviation of each observation
        """
        return numpy.sqrt(self.cd.variance())


    def covariance(self):
        """
        Cd, dense; for the analyses of small problems
        """
        return self.cd.dense()


    def _whiten(self, prediction, batch):
        """
        The first {batch} rows of {prediction} whitened, on the host
        """
        rows = numpy.asarray(self._grid(prediction))
        rows[:batch] = self.cd.whiten(rows[:batch])
        return self


# end of file
