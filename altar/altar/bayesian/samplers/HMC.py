# -*- python -*-
# -*- coding: utf-8 -*-
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
class HMC(altar.component, family="altar.samplers.hmc", implements=sampler):
    """
    Hamiltonian Monte Carlo: propose a candidate by simulating (leapfrog discretized)
    Hamiltonian dynamics from a freshly sampled momentum, then accept/reject with the usual
    Metropolis-Hastings criterion in energy space.

    I am the one piece pyre ever registers and a {.pfg} ever names. My own job is small: pick
    my backend implementation once, in {initialize}, and forward every protocol method to it
    from then on -- no backend check anywhere outside this one place; the same shape
    {altar.bayesian.samplers.Metropolis}/{altar.distributions.Base}/etc. already use.

    My implementation lives in a same-named class in {altar.bayesian.samplers.native} (the
    cpu default) or {altar.bayesian.samplers.cuda}. Neither is a pyre component; each is a
    plain class holding the actual algorithm. Only the cuda implementation currently supports
    reparameterized models (via {model.reparameterization}); the cpu one does not, and takes
    the traits below unchanged.
    """

    # user configurable state
    leapfrog_steps = altar.properties.int(default=10)
    leapfrog_steps.doc = "the number of leapfrog substeps per trajectory"

    step_size = altar.properties.float(default=0.01)
    step_size.doc = "the leapfrog step size epsilon; adapted after each trajectory by {stepsizer}"

    # step size regulator
    stepsizer = altar.bayesian.stepsizer()
    stepsizer.doc = "the step size regulator that adjusts {step_size} based on acceptance statistics"

    adapt_mass_matrix = altar.properties.bool(default=True)
    adapt_mass_matrix.doc = \
        "whether to precondition the leapfrog dynamics with a diagonal mass matrix estimated " \
        "from the current population, instead of a fixed unit mass"

    mass_update_interval = altar.properties.int(default=20)
    mass_update_interval.doc = "how often, in trajectories, to re-estimate the mass matrix"

    min_variance = altar.properties.float(default=1e-8)
    min_variance.doc = "lower bound on the per-parameter variance used to build the mass matrix"

    max_variance = altar.properties.float(default=1e8)
    max_variance.doc = "upper bound on the per-parameter variance used to build the mass matrix"


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
        self._impl.leapfrog_steps = self.leapfrog_steps
        self._impl.step_size = self.step_size
        self._impl.stepsizer = self.stepsizer
        self._impl.adapt_mass_matrix = self.adapt_mass_matrix
        self._impl.mass_update_interval = self.mass_update_interval
        self._impl.min_variance = self.min_variance
        self._impl.max_variance = self.max_variance
        # let it initialize itself
        self._impl.initialize(application=application)
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


    @altar.export
    def update(self, annealer, statistics):
        """
        Notification that a β step is complete; a no-op, forwarded for protocol conformance --
        both backends already adjust their step size per-trajectory, inside {sample_posterior}
        """
        return self._impl.update(annealer=annealer, statistics=statistics)


    # public data
    @property
    def scaling(self):
        """
        {Annealer} logs/archives a generic "scaling" value after every β step, reaching
        directly into {self.scaling} -- unlike {Metropolis}, my step size changes every
        trajectory, not just at {update} boundaries, so this is a live read of my
        implementation's current value, not a snapshot copied on some specific call
        """
        return self._impl.step_size


    # private data
    _impl = None # my backend implementation, chosen once, in {initialize}


# end of file
