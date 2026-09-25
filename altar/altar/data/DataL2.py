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
from .Base import Base as base


# declaration
class DataL2(base, family="altar.data.datal2"):
    """
    The observed data with L2 norm

    My actual numerics live in {altar.data.native.DataL2.DataL2} (cpu) or
    {altar.data.cuda.DataL2.DataL2} (cuda); see {Base} for how one gets picked.
    """

    data_file = altar.properties.path(default="data.txt")
    data_file.doc = "the name of the file with the observations"

    observations = altar.properties.int(default=1)
    observations.doc = "the number of observed data"

    cd_file = altar.properties.path(default=None)
    cd_file.doc = "the name of the file with the data covariance matrix"

    cd_std = altar.properties.float(default=1.0)
    cd_std.doc = "the constant covariance for data, sigma^2"

    cd_dtype = altar.properties.str(default=None)
    cd_dtype.doc = "the data type (float32/float64) for Cd computations, if different from " \
                   "the job's own precision; cuda only"

    merge_cd_with_data = altar.properties.bool(default=False)
    merge_cd_with_data.doc = "whether to merge cd with data; cpu only, the cuda " \
                              "implementation always merges"

    norm = altar.norms.norm()
    norm.default = altar.norms.l2()
    norm.doc = "the norm to use when computing the data log likelihood"


    # implementation details
    def _makeImpl(self):
        """
        Hand my implementation my own extra state, beyond what {Base} already copies over
        """
        impl = super()._makeImpl()
        impl.data_file = self.data_file
        impl.observations = self.observations
        impl.cd_file = self.cd_file
        impl.cd_std = self.cd_std
        impl.cd_dtype = self.cd_dtype
        impl.merge_cd_with_data = self.merge_cd_with_data
        impl.norm = self.norm
        return impl


# end of file
