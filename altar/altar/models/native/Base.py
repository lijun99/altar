# -*- python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#


# the declaration
class Base:
    """
    Shared support for the cpu implementation of a parameter set: a plain class, not a pyre
    component -- the registered component is the shim in {altar.models} that owns {count} as
    a configurable trait and copies its value down to me, once, at {initialize} time.

    None of these are ever overridden on the cpu side today -- they exist so the shim's
    generic forwarding always has something to call, whichever backend is active.
    """


    def constrain(self, theta, batch=None):
        """
        Force the samples in {theta} back within my constraints, in place. A cpu sampler
        rejects through {verify} instead, so there is nothing to do here.
        """
        return self


    def eval_prior_with_physical(self, theta, prior, batch=None):
        """
        Add any prior contributions that depend on physical parameters, beyond what
        {eval_prior} already contributed. Default: nothing further to add.
        """
        return self


    def eval_prior_physical(self, theta, prior, batch=None):
        """
        Fill {prior} with the log likelihoods of the samples in {theta}, given in physical
        space. Without reparameterization, physical space is sampling space, so the default
        is just {eval_prior}.
        """
        return self.eval_prior(theta=theta, prior=prior, batch=batch)


    def jacobian(self, theta, jacobian, batch=None):
        """
        Fill {jacobian} with d(physical)/d(sampling). Default: nothing to do -- {jacobian}
        is expected to already hold 1, the correct value for an unreparameterized parameter
        set.
        """
        return self


    def to_physical(self, theta, batch=None):
        """
        Transform {theta} from sampling space to physical space, in place. Without
        reparameterization, the two coincide, so the default is a no-op.
        """
        return self


    def to_sampling(self, theta, batch=None):
        """
        Transform {theta} from physical space to sampling space, in place; the default is
        likewise a no-op.
        """
        return self


    # implementation details
    def restrict(self, theta):
        """
        Return my portion of the sample matrix {theta}
        """
        # find out how many samples in the set
        samples = theta.rows
        # find where my samples live within the overall sample matrix, and how wide my slice is
        start = 0, self.offset
        shape = samples, self.count
        # return a view to the portion of the sample that's mine
        return theta.view(start=start, shape=shape)


    # private data, set by the shim before any other method runs
    count = None
    offset = 0


# end of file
