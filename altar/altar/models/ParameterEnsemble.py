# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-2021 parasim inc
# (c) 2010-2021 california institute of technology
# all rights reserved
#
# Author(s): Lijun Zhu

# the package
import altar
# the protocol
from .ParameterSet import ParameterSet as parameters
# my base class
from .Base import Base as base


# component
class ParameterEnsemble(base, family="altar.models.parameters.parameterensemble"):
    """
    An ensemble of parameter sets

    My actual numerics live in {altar.models.native.ParameterEnsemble.ParameterEnsemble} (cpu)
    or {altar.models.cuda.ParameterEnsemble.ParameterEnsemble}; see {Base} for how one gets
    picked.
    """

    # user configurable state
    count = altar.properties.int(default=0)
    count.doc = "the total number of parameters in this ensemble"

    prior = altar.distributions.distribution(default=None)
    prior.doc = "not used; provided for protocol compatibility"

    prep = altar.distributions.distribution(default=None)
    prep.doc = "not used; provided for protocol compatibility"

    psets_list = altar.properties.list(schema=altar.properties.str(), default=None)
    psets_list.doc = "list of parameter set names that defines the order"

    psets = altar.properties.dict(schema=parameters())
    psets.default = dict()
    psets.doc = "the collection of parameter sets in the ensemble"


    # implementation details
    def _makeImpl(self):
        """
        Hand my implementation my own extra state, beyond what {Base} already copies over
        """
        impl = super()._makeImpl()
        impl.psets = self.psets
        impl.psets_list = self.psets_list
        return impl


# end of file
