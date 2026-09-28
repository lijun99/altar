# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# externals
import numpy
# the package
import altar
# my base class
from altar.models.linear.Linear import Linear


# declaration
class Static(Linear, family="altar.models.seismic.static"):
    """
    Static slip inversion, d = G theta: the linear model, with the strike and dip slips of
    {patches} fault patches as its parameters

    Everything else -- both backends, the gradient, the forward problem -- is the linear model's;
    what I add is a model uncertainty C_p from the uncertain earth model the green's functions
    were computed in, see {compute_cp}
    """


    # user configurable state
    patches = altar.properties.int(default=None)
    patches.doc = "the number of fault patches, each with a strike and a dip slip"

    green = altar.properties.path(default="static.gf.h5")
    green.doc = "the green's functions, (observations, 2*patches)"

    # the inputs for estimating the model uncertainty C_p, with cp=altar.models.cp.adaptive
    cmu_file = altar.properties.path(default="static.Cmu.h5")
    cmu_file.doc = "C_mu, the covariance of the uncertain earth model inputs mu, (n x n)"

    kmu_file = altar.properties.path(default="static.kernel.h5")
    kmu_file.doc = "the sensitivity kernels dG/dmu_i, one (observations x 2*patches) dataset each"


    # protocol obligations
    @altar.export
    def initialize(self, application):
        """
        The linear model's setup, plus a check that my parameters match my patches
        """
        super().initialize(application=application)
        if self.patches is not None and self.parameters != 2 * self.patches:
            self.error.log(
                f"the static model with {self.patches} patches needs {2 * self.patches} "
                f"parameters, but its parameter sets hold {self.parameters}")
            raise SystemExit(1)
        return self


    def compute_cp(self, theta):
        """
        C_p = K_p C_mu K_p^T for the mean model {theta}, K_p[:, i] = K_i theta
        """
        from altar.models.cp import sensitivity
        θ = numpy.asarray(theta, dtype=float)
        return sensitivity(model=self, cmu_file=self.cmu_file, kmu_file=self.kmu_file,
                           predict=lambda kernel: kernel.reshape(self.observations, self.parameters) @ θ)


# end of file
