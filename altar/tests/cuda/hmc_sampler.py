#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#


"""
Sanity check: {altar.bayesian.samplers.cuda.HMC}, against a real GPU -- the full Hamiltonian
Monte Carlo sampler, end to end: protocol conformance ({initialize}/{sample_posterior}/
{update}, matching {Sampler}, which this class did not satisfy before this session's fix --
see the class docstring and altar/altar/bayesian/samplers/__init__.py's {hmc()} foundry),
state allocation, leapfrog integration (position/momentum updates via the ported
{altar.cuda.libcudaaltar.leapfrog} kernels), potential/gradient evaluation, kinetic energy,
and the Metropolis-Hastings accept/reject decision.

The target is a standard normal (prior = -0.5*||theta||^2, no data term): starting the chains
from near a point mass and running enough trajectories, the population's spread should
equilibrate to std ~= 1 with a high acceptance rate throughout, since HMC's leapfrog
trajectory conserves energy well at a modest step size on such an easy target. This is a
statistical sanity check (not a strict numerical assertion), the same spirit as the cpu
{HMC} sampler's own validation.
"""


def test():
    import numpy
    import altar
    from altar.bayesian.samplers.cuda.HMC import HMC
    from altar.bayesian.stepsizers.StepSizer import FixedStepSize

    samples = 4000
    nparams = 3

    class _Model:
        parameters = nparams
        reparameterization = False

        def likelihoods(self, annealer, step):
            arr = numpy.asarray(step.theta)
            numpy.asarray(step.prior)[:] = -0.5 * numpy.sum(arr**2, axis=1)
            numpy.asarray(step.data)[:] = 0.0

        def gradient(self, controller, step, batch=None):
            arr = numpy.asarray(step.theta)
            numpy.asarray(step.prior_gradient)[:, :] = -arr
            numpy.asarray(step.data_gradient)[:, :] = 0.0

    class _Dispatcher:
        sample_posterior_start = "sp_start"
        sample_posterior_finish = "sp_finish"
        chain_advance_start = "ca_start"
        chain_advance_finish = "ca_finish"

        def notify(self, event, controller):
            return

    class _Job:
        chains = samples
        gpuprecision = "float64"
        steps = 1  # one trajectory per call to sample_posterior; the test itself supplies
                   # the repetition, via 150 external calls below

    class _Info:
        @staticmethod
        def log(*args, **kwds):
            return

    class _Application:
        model = _Model()
        job = _Job()
        info = _Info()

    class _Annealer:
        model = _Model()
        dispatcher = _Dispatcher()

    class _Step:
        def __init__(self, rng):
            self.beta = 1.0
            self.theta = altar.matrix(shape=(samples, nparams))
            numpy.asarray(self.theta)[:, :] = rng.normal(size=(samples, nparams)) * 0.01
            self.momentum = None
            self.prior = altar.vector(shape=samples)
            self.data = altar.vector(shape=samples)
            self.posterior = altar.vector(shape=samples)
            self.U = altar.vector(shape=samples)
            self.H = altar.vector(shape=samples)
            self.prior_gradient = altar.matrix(shape=(samples, nparams))
            self.data_gradient = altar.matrix(shape=(samples, nparams))
            self.U_gradient = altar.matrix(shape=(samples, nparams))

    sampler = HMC()  # a plain impl class now; the pyre component is the shim,
                     # {altar.bayesian.samplers.HMC}, which this test bypasses
    # bypassing the shim's initialize() (and its {_makeImpl} copy-down), so attach the
    # component-typed state it would normally hand me before calling my own initialize();
    # a fixed (non-adaptive) step size, since this test's target is the leapfrog kernels
    # themselves at a fixed, hand-picked-good step, not the stepsizer's own convergence
    sampler.stepsizer = FixedStepSize()
    sampler.initialize(application=_Application())
    assert sampler.proposal_state is not None
    # 20 leapfrog substeps per trajectory, a larger-than-default step size (both previously
    # set via magic attributes on {step} that nothing in the real pipeline ever set -- see
    # {altar.bayesian.samplers.cuda.HMC._walk}, which now reads {self.leapfrog_steps}/
    # {self.step_size} directly instead)
    sampler.leapfrog_steps = 20
    sampler.step_size = 0.15

    step = _Step(numpy.random.default_rng(0))
    annealer = _Annealer()

    total_accepted = total_attempts = 0
    for _ in range(150):
        stats = sampler.sample_posterior(annealer=annealer, step=step)
        total_accepted += stats.accepted
        total_attempts += stats.accepted + stats.rejected
        assert stats.invalid == 0  # hmc has no notion of an invalid candidate

    arr = numpy.asarray(step.theta)
    assert abs(arr.mean()) < 0.1
    assert abs(arr.std() - 1.0) < 0.1
    assert total_accepted / total_attempts > 0.9  # leapfrog conserves energy well here

    # update() is a no-op: the step size is already adjusted per-trajectory in _walk
    assert sampler.update(annealer=annealer, statistics=stats) is None

    # all done
    return


# main
if __name__ == "__main__":
    test()


# end of file
