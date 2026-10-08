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

if typing.TYPE_CHECKING:
    from altar.models.BayesianL2 import BayesianL2
    from altar.shells.Application import Application

# the declaration
class Base:
    """
    Shared support for the cpu implementation of a parameter set: a plain class, not a pyre
    component -- the registered component is the shim in {altar.models} that owns {count} as
    a configurable trait and copies its value down to me, once, at {initialize} time.

    None of these are ever overridden on the cpu side today -- they exist so the shim's
    generic forwarding always has something to call, whichever backend is active.
    """


    def constrain(self, theta: numpy.ndarray, batch: int | None = None) -> typing.Self:
        """
        Force the samples in {theta} back within my constraints, in place. A cpu sampler
        rejects through {verify} instead, so there is nothing to do here.
        """
        return self


    def eval_prior_with_physical(self, theta: numpy.ndarray, prior: numpy.ndarray,
                                 batch: int | None = None) -> typing.Self:
        """
        Add any prior contributions that depend on physical parameters, beyond what
        {eval_prior} already contributed. Default: nothing further to add.
        """
        return self


    def eval_prior_physical(self, theta: numpy.ndarray, prior: numpy.ndarray,
                            batch: int | None = None) -> typing.Self:
        """
        Fill {prior} with the log likelihoods of the samples in {theta}, given in physical
        space. Without reparameterization, physical space is sampling space, so the default
        is just {eval_prior}.
        """
        return self.eval_prior(theta=theta, prior=prior, batch=batch)


    def jacobian(self, theta: numpy.ndarray, jacobian: numpy.ndarray,
                 batch: int | None = None) -> typing.Self:
        """
        Fill {jacobian} with d(physical)/d(sampling). Default: nothing to do -- {jacobian}
        is expected to already hold 1, the correct value for an unreparameterized parameter
        set.
        """
        return self


    def to_physical(self, theta: numpy.ndarray, batch: int | None = None) -> typing.Self:
        """
        Transform {theta} from sampling space to physical space, in place. Without
        reparameterization, the two coincide, so the default is a no-op.
        """
        return self


    def to_sampling(self, theta: numpy.ndarray, batch: int | None = None) -> typing.Self:
        """
        Transform {theta} from physical space to sampling space, in place; the default is
        likewise a no-op.
        """
        return self


    # implementation details
    def restrict(self, theta: numpy.ndarray) -> numpy.ndarray:
        """
        Return my portion of the sample matrix {theta}
        """
        # a view of my columns
        return theta[:, self.offset:self.offset + self.count]


    # private data, set by the shim before any other method runs
    count: int
    offset: int = 0


# end of file
