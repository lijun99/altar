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
class NoCp(altar.component, family="altar.models.cp.none", implements=Cp):
    """
    No model uncertainty: C_chi = C_d
    """


    @altar.export
    def initialize(self, model, application):
        """
        Nothing to set up
        """
        return self


    @altar.export
    def update(self, model, annealer, step):
        """
        Nothing ever changes
        """
        return False


    @altar.export
    def apply(self, model, theta):
        """
        Nothing to add
        """
        return self


# end of file
