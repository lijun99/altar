# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# the package
import altar
# my protocol
from .Cp import Cp


# declaration
class Fixed(altar.component, family="altar.models.cp.fixed", implements=Cp):
    """
    A user-supplied C_p, added to C_d once, before sampling starts
    """


    # user configurable state
    cp_file = altar.properties.path(default="cp.h5")
    cp_file.doc = "the (observations x observations) C_p, among the model's input files"

    dataset = altar.properties.str(default=None)
    dataset.doc = "the dataset holding C_p in an .h5 {cp_file}; default the first one"


    @altar.export
    def initialize(self, model, application):
        """
        Load C_p and fold it into the model's data covariance
        """
        cp = model.io.load(filename=self.cp_file, shape=(model.observations, model.observations),
                           dataset=self.dataset, dtype="float64")
        model.update_covariance(cp=cp)
        return self


    @altar.export
    def update(self, model, annealer, step):
        """
        C_p is fixed
        """
        return False


    @altar.export
    def apply(self, model, theta):
        """
        C_p is already part of the data covariance
        """
        return self


# end of file
