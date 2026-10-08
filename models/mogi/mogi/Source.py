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
from math import sqrt, pi as π
import numpy


# declaration
class Source:
    """
    An implementation of Mogi[1958]

    The surface displacement calculation for a point pressure source in an elastic half space.
    """


    # public data
    # location
    x: float = 0
    y: float = 0
    d: float = 0
    # strength
    dV: float = 0
    # material properties
    nu: float = .25 # Poisson ration


    # interface
    def displacements(self, locations: typing.Sequence[tuple[float, float]],
                      los: numpy.ndarray) -> numpy.ndarray:
        """
        Compute the expected displacements from a point pressure source at a set of observation
        locations
        """
        # the location of the source
        x_src = self.x
        y_src = self.y
        d_src = self.d
        # its strength
        dV = self.dV
        # get the material properties
        nu = self.nu

        # allocate space for the result
        u = numpy.zeros(len(locations))
        # go through each observation location
        for index, (x_obs,y_obs) in enumerate(locations):
            # compute displacements
            x = x_src - x_obs
            y = y_src - y_obs
            d = d_src
            # compute the distance to the point source
            x2 = x**2
            y2 = y**2
            d2 = d**2
            # intermediate values
            C = (nu-1) * dV/π
            R = sqrt(x2 + y2 + d2)
            CR3 = C * R**-3
            # store the expected displacement
            u[index] = x*CR3 * los[index, 0] + y*CR3 * los[index, 1] - d*CR3 * los[index,2]

        # all done
        return u


    # meta-methods
    def __init__(self, x: float = x, y: float = y, d: float = d, dV: float = dV, nu: float = nu,
                 **kwds) -> None:
        # chain up
        super().__init__(**kwds)
        # store the location
        self.x = x
        self.y = y
        self.d = d
        # the strength
        self.dV = dV
        # and the material properties
        self.nu = nu
        # all done
        return


# end of file
