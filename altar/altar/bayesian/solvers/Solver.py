# -*- python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#


# externals
from __future__ import annotations
import typing
import numpy
# get the package
import altar

if typing.TYPE_CHECKING:
    from altar.bayesian.schedulers.COV import COV
    from altar.shells.Application import Application


# the scheduler protocol
class Solver(altar.protocol, family="altar.bayesian.solvers"):
    """
    The protocol that all δβ solvers must implement
    """


    # user configurable state
    tolerance = altar.properties.float()
    tolerance.doc = 'the fractional tolerance for achieving convergence'


    # required behavior
    @altar.provides
    def initialize(self, application: Application, scheduler: COV) -> typing.Self:
        """
        Initialize me and my parts given an {application} context and a {scheduler}
        """


    @altar.provides
    def solve(self, llk: numpy.ndarray, weight: numpy.ndarray) -> tuple[float, float]:
        """
        Compute the next temperature in the cooling schedule, and the COV of the normalized
        weights it gives {llk}, which are left in {weight}
        """


    # framework hooks
    @classmethod
    def pyre_default(cls, **kwds) -> type:
        """
        Provide a default implementation
        """
        # by default, find the root with Brent's method
        from .Brent import Brent
        # and return it
        return Brent


# end of file
