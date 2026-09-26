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
    Shared support for the cuda implementation of a parameter set: a plain class, not a pyre
    component -- the registered component is the shim in {altar.models} that owns {count} as
    a configurable trait and copies its value down to me, once, at {initialize} time.

    Unlike the cpu side, a cuda parameter set never restricts {theta} to its own slice before
    delegating to its distributions -- each distribution already knows its own slice (its
    {idx_range}) and reads it directly out of the full {theta}, so these methods just pass
    {theta} straight through.
    """


    def constrain(self, theta, batch=None):
        """
        Force the samples in {theta} back within my constraints, in place. Default: nothing
        to do.
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
        is expected to already hold 1 (see {altar.bayesian.states.cuda.HMCState.alloc}'s own
        default fill), the correct value for an unreparameterized parameter set.
        """
        return self


    def to_physical(self, theta, batch=None):
        """
        Transform {theta} from sampling space to physical space, in place. Default: a no-op.
        """
        return self


    def to_sampling(self, theta, batch=None):
        """
        Transform {theta} from physical space to sampling space, in place. Default: a no-op.
        """
        return self


    # private data, set by the shim before any other method runs
    count = None
    offset = 0


# end of file
