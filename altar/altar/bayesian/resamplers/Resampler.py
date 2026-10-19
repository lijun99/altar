# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#


# externals
from __future__ import annotations
import typing
# the package
import altar

if typing.TYPE_CHECKING:
    import numpy
    from altar.shells.Application import Application


# declaration
class Resampler(altar.protocol, family="altar.bayesian.resamplers"):
    """
    The protocol of the components that replace a weighted sample by an equally weighted one
    """


    # required behavior
    @altar.provides
    def initialize(self, application: Application) -> typing.Self:
        """
        Initialize me given an {application} context
        """

    @altar.provides
    def multiplicities(self, w: numpy.ndarray) -> numpy.ndarray:
        """
        How many copies of each sample to keep, given their normalized weights {w}
        """

    @altar.provides
    def rows(self, w: numpy.ndarray) -> numpy.ndarray:
        """
        The sample behind each one of the new sample, in random order, given the weights {w}
        """


    # framework hooks
    @classmethod
    def pyre_default(cls, **kwds) -> type:
        """
        Supply a default implementation
        """
        from .ImportanceResampler import ImportanceResampler
        return ImportanceResampler


# end of file
