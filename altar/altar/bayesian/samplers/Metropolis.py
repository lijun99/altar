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
from importlib import import_module
# the package
import altar
# my protocol
from .Sampler import Sampler as sampler


# declaration
class Metropolis(altar.component, family="altar.samplers.metropolis", implements=sampler):
    """
    The Metropolis algorithm as a sampler of the posterior distribution.

    I am the one piece pyre ever registers and a {.pfg} ever names. My own job is small: pick
    my backend implementation once, in {initialize}, and forward every protocol method to it
    from then on -- no backend check anywhere outside this one place. This is the same shape
    {altar.distributions.Base}/{altar.models.Base}/etc. already use; see their docstrings.

    My implementation lives in a same-named class in {altar.bayesian.samplers.native} (the
    cpu default) or {altar.bayesian.samplers.cuda}. Neither is a pyre component; each is a
    plain class holding the actual algorithm.
    """

    # proposal mechanism (optional depending on sampler)
    proposal = altar.bayesian.proposal()
    proposal.doc = "the proposal mechanism used by this sampler"

    # step size regulator
    stepsizer = altar.bayesian.stepsizer()
    stepsizer.doc = "the step size regulator that adjusts the proposal scaling based on acceptance statistics"

    # step count regulator: how many MC steps to run per β step
    stepcounter = altar.bayesian.stepcounter()
    stepcounter.doc = "the step count regulator that decides how many MC steps to run per β step"

    # the initial proposal scaling; {stepsizer} adjusts it after every β step
    scaling = altar.properties.float(default=.1)
    scaling.doc = "the parameter covariance Σ is scaled by the square of this"


    # protocol obligations
    @altar.export
    def initialize(self, application):
        """
        Pick my backend implementation and let it initialize itself given an {application}
        context
        """
        # pick my backend implementation, once
        self._impl = self._makeImpl()
        # hand it the state every backend shares
        self._impl.proposal = self.proposal
        self._impl.stepsizer = self.stepsizer
        self._impl.stepcounter = self.stepcounter
        self._impl.scaling = self.scaling
        # let it initialize itself
        self._impl.initialize(application=application)
        # mirror the (possibly stepsizer-adjusted) initial scaling back onto me: {Annealer}
        # reads {self.scaling} directly (see {update}), not {self._impl.scaling}
        self.scaling = self._impl.scaling
        # all done
        return self

    def _makeImpl(self):
        """
        Build my backend implementation: a same-named class in {native} (the cpu default) or
        {cuda}, picked once, here, based on {altar.backends.active()}
        """
        # my own class name is also my implementation's
        name = type(self).__name__
        # the package it lives in
        backend = "cuda" if altar.backends.active() == "cuda" else "native"
        # reach it
        module = import_module(f"altar.bayesian.samplers.{backend}.{name}")
        factory = getattr(module, name)
        # build and return it
        return factory()


    @altar.export
    def sample_posterior(self, annealer, step):
        """
        Sample the posterior distribution
        """
        return self._impl.sample_posterior(annealer=annealer, step=step)


    def restore(self, scaling):
        """
        Continue with the proposal {scaling} an earlier run reached
        """
        # let the step size regulator start from it
        self.scaling = self.stepsizer.initialize(value=scaling)
        # and my implementation walk with it
        self._impl.scaling = self.scaling
        # all done
        return self


    @altar.export
    def update(self, annealer, statistics):
        """
        Update my parameters based on the results of walking my Markov chains
        """
        # delegate to my implementation
        self._impl.update(annealer=annealer, statistics=statistics)
        # and mirror the adjusted scaling back onto me; {Annealer.py} reads {self.scaling}
        # directly, right after calling this
        self.scaling = self._impl.scaling
        # all done
        return


    # private data
    _impl = None # my backend implementation, chosen once, in {initialize}


# end of file
