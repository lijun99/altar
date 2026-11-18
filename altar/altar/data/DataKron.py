# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# the package
import altar
# my base class
from .DataL2 import DataL2


# declaration
class DataKron(DataL2, family="altar.data.datakron"):
    """
    The observed data with L2 norm and a covariance separable in space and time, per component,
    Cd = blockdiag_c(S_c ⊗ R_c), see {altar.data.Kronecker}: the observations ordered as
    (component, station, bin), the bin fastest

    The data are whitened with the cholesky factors of S_c and R_c, without forming Cd; models
    see in {cd_inv} a factor L, Cd^-1 = L L^T, that multiplies rows on the right, x @ L, and
    whitens their green's functions with it once. No mask, and no C_p
    """

    noise_file = altar.properties.path(default="noise.h5")
    noise_file.doc = "the h5 file with S, (components, stations, stations), the covariance of the " \
                     "stations, and R, (components, bins, bins), that of the bins"

    spatial_dataset = altar.properties.str(default="S")
    spatial_dataset.doc = "the name of the dataset of S in {noise_file}"

    temporal_dataset = altar.properties.str(default="R")
    temporal_dataset.doc = "the name of the dataset of R in {noise_file}"


    # implementation details
    def _makeImpl(self):
        """
        Hand my implementation my own extra state
        """
        impl = super()._makeImpl()
        impl.noise_file = self.noise_file
        impl.spatial_dataset = self.spatial_dataset
        impl.temporal_dataset = self.temporal_dataset
        return impl


# end of file
