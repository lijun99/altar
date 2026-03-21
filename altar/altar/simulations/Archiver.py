# -*- python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
#
# (c) 2013-2021 parasim inc
# (c) 2010-2021 california institute of technology
# all rights reserved
#

# the package
import altar

# the archiver protocol
class Archiver(altar.protocol, family="altar.simulations.archivers"):
    """
    The protocol that all AlTar simulation archivers must implement

    Archivers persist intermediate simulation state and can be used to restart a simulation
    """

    # required behavior
    @altar.provides
    def initialize(self, application):
        """
        Initialize me given an {application} context
        """

    @altar.provides
    def record(self, step):
        """
        Record the final state of the simulation
        """

    @altar.provides
    def write(self, path, data, info=None):
        """
        Persist one dataset.

        {path} is a slash-separated string "Group/Subgroup/Name"; a bare name with no
        slash is stored at the top level.  {data} may be any of:
          - an object with a .ndarray() method  (altar.matrix, altar.vector, gsl objects)
          - a numpy ndarray
          - a Python scalar

        {info} is an optional dict of metadata (written as HDF5 attributes or stored
        alongside the data in other backends).
        """

    @altar.provides
    def register(self, component):
        """
        Register {component} so that its record(archiver) method is called at each
        save point.  Components call this during their own initialize().
        """

    # framework hooks
    @classmethod
    def pyre_default(cls, **kwds):
        """
        Supply a default implementation
        """
        # pull the in-memory archiver
        from .Recorder import Recorder as default
        # and return it
        return default

# end of file
