# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# get the package
import altar
# and my base class
from .Base import Base as base


# the declaration
class Preset(base, family="altar.distributions.preset"):
    """
    Samples read from a file, e.g. the posterior of an earlier run; for initializing a parameter
    set (as its {prep}) only, never as a prior

    My actual numerics live in {altar.distributions.native.Preset.Preset} (cpu) or
    {altar.distributions.cuda.Preset.Preset}; see {Base} for how one gets picked.
    """


    # user configurable state
    input_file = altar.properties.path(default=None)
    input_file.doc = "an .h5 file with the samples, e.g. an archived step; relative to the " \
                     "current directory, or else among the model's input files"

    dataset = altar.properties.str(default=None)
    dataset.doc = "the (samples x parameters) dataset, e.g. ParameterSets/strikeslip; for an " \
                  "archived step, its _physical, or else _sampling, samples are used"


    # implementation details
    def _makeImpl(self):
        """
        Hand my implementation my own extra state, beyond what {Base} already copies over
        """
        impl = super()._makeImpl()
        impl.input_file = self.input_file
        impl.dataset = self.dataset
        return impl


# end of file
