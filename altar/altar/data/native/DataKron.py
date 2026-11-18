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
import numpy
# my base class
from .DataL2 import DataL2
from ..Kronecker import Kronecker


class DataKron(DataL2):
    """
    The cpu implementation of observed data with a {Kronecker} covariance: {cd} is the
    covariance, {cd_inv} its factor L, Cd^-1 = L L^T, applied as x @ L
    """

    def load_data(self) -> None:
        """
        The observations and the factors of their covariance
        """
        self.cd = load_kronecker(self)
        self._observed = self.io.load(
            filename=self.data_file, shape=self.observations, dataset=self.datafile_dataset)
        if not numpy.isfinite(self._observed).all():
            self.error.log(f"non-finite values in '{self.data_file}'")
            raise SystemExit(1)
        self.mask = None
        self._raw = self._observed.astype(self.precision)
        self.dataobs = self._raw.copy()
        return


    def initialize_covariance(self, cd: Kronecker) -> DataKron:
        """
        The normalization, the factor L of Cd^-1, and the data whitened by it, if asked to
        """
        self.normalization = -0.5 * (self.observations * math.log(2 * math.pi) + cd.logdet)
        self.cd_inv = cd.factor
        if self.merge_cd_with_data:
            self.dataobs = cd.whiten(self._observed).astype(self.precision)
        return self


    def update_covariance(self, cp: numpy.ndarray | None = None) -> DataKron:
        """
        Back to Cd; a C_p would break the separable structure
        """
        if cp is not None:
            self.error.log("datakron: a C_p needs a dense covariance, altar.data.datal2")
            raise SystemExit(1)
        return self.initialize_covariance(cd=self.cd)


    def sigma(self) -> numpy.ndarray:
        """
        The standard deviation of each observation
        """
        return numpy.sqrt(self.cd.variance())


    def covariance(self) -> numpy.ndarray:
        """
        Cd, dense; for the analyses of small problems
        """
        return self.cd.dense()


def load_kronecker(data) -> Kronecker:
    """
    The {Kronecker} covariance of the configuration of {data}, checked against its observations
    """
    if data.mask_dataset is not None or data.cd_file is not None:
        data.error.log("datakron: its covariance is in noise_file, and it takes no mask; fill or "
                       "drop the missing bins first")
        raise SystemExit(1)
    import h5py
    try:
        path = data.ifs[data.noise_file].uri.path
    except data.ifs.NotFoundError:
        data.error.log(f"missing input: no '{data.noise_file}' in '{data.ifs.path()}'")
        raise SystemExit(1)
    with h5py.File(path, "r") as h5:
        S, R = numpy.asarray(h5[data.spatial_dataset]), numpy.asarray(h5[data.temporal_dataset])
    try:
        cd = Kronecker(S, R)
    except (ValueError, numpy.linalg.LinAlgError) as error:
        data.error.log(f"datakron: '{data.noise_file}': {error}")
        raise SystemExit(1)
    if cd.observations != data.observations:
        data.error.log(f"datakron: {cd.components} components x {cd.stations} stations x {cd.bins} "
                       f"bins are {cd.observations} observations, not {data.observations}")
        raise SystemExit(1)
    data.info.log(f"datakron: {cd.components} components, {cd.stations} stations, {cd.bins} bins")
    return cd


# end of file
