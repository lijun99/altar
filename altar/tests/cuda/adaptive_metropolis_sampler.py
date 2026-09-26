#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#


"""
Sanity check: {altar.bayesian.samplers.cuda.Metropolis} configured with
{altar.bayesian.stepcounters.DecorrelatingSteps}, against a real GPU.

{AdaptiveMetropolis} is gone -- the decorrelation-based outer loop it used to hardcode is now
just the generic {stepcounter} protocol's {DecorrelatingSteps} implementation, so the same
{Metropolis} class covers both the old fixed-step and adaptive behaviors via configuration.
{prepare_sampling_pdf}/{finish_sampling_pdf}/{displace}/{allocate_gpu_data}/{sample_posterior}/
{update} and the {proposal}/{stepsizer} components are unchanged; only {stepcounter} is swapped
from the default {FixedSteps} to {DecorrelatingSteps}.
"""


def test():
    import numpy
    import altar
    from altar.bayesian.samplers.cuda.Metropolis import Metropolis
    from altar.bayesian.stepsizers.StepSizer import TargetedRate
    from altar.bayesian.stepcounters.StepCounter import DecorrelatingSteps

    samples, parameters = 500, 3

    class _Info:
        def log(self, message):
            return

    class _Dispatcher:
        def __getattr__(self, name):
            return name

        def notify(self, event, controller):
            return

    class _Worker:
        workers = 1

    class _Model:
        def verify_theta(self, theta, mask, batch):
            # standard normal: unbounded support, every candidate valid
            return mask

        def likelihoods(self, annealer, step, batch):
            arr = numpy.asarray(step.theta)[:batch]
            numpy.asarray(step.prior)[:batch] = -0.5 * numpy.sum(arr**2, axis=1)
            numpy.asarray(step.data)[:batch] = 0.0
            numpy.asarray(step.posterior)[:batch] = (
                numpy.asarray(step.prior)[:batch] + step.beta * numpy.asarray(step.data)[:batch]
            )

    class _Annealer:
        model = _Model()
        dispatcher = _Dispatcher()
        worker = _Worker()

    sampler = Metropolis()  # a plain impl class now; the pyre component is the shim,
                            # {altar.bayesian.samplers.Metropolis}, which this test bypasses
    sampler.curng = altar.cuda.curand_generator()
    sampler.precision = "float64"
    # bypassing the shim's initialize() (and its {_makeImpl} copy-down), so attach the
    # component-typed state it would normally hand me, and fill in the target my own
    # initialize() would have (random-walk Metropolis's theoretically-optimal acceptance rate)
    sampler.stepsizer = TargetedRate()
    sampler.stepsizer.target = 0.234
    sampler.scaling = sampler.stepsizer.initialize(value=0.5)
    # swap in the decorrelation-based step counter; relax its thresholds so the test exits
    # quickly
    sampler.stepcounter = DecorrelatingSteps(name="decorrelating-test")
    sampler.stepcounter.info = _Info()
    sampler.stepcounter.min_mc_steps = 5
    sampler.stepcounter.corr_check_steps = 5
    sampler.stepcounter.max_mc_steps = 10
    sampler.stepcounter.max_mc_steps_stage2 = 10
    sampler.stepcounter.target_correlation = 0.999

    sampler.allocate_gpu_data(samples, parameters)
    numpy.asarray(sampler.gsigma_chol)[:, :] = numpy.eye(parameters)
    sampler.gsigma_chol.cholesky()
    sampler.gsigma_chol *= sampler.scaling

    step = sampler.CoolingStep.alloc(samples, parameters, dtype="float64")
    rng = numpy.random.default_rng(4)
    numpy.asarray(step.theta)[:, :] = rng.normal(size=(samples, parameters))
    numpy.asarray(step.prior)[:] = -0.5 * numpy.sum(numpy.asarray(step.theta) ** 2, axis=1)
    numpy.asarray(step.data)[:] = 0.0
    numpy.asarray(step.posterior)[:] = numpy.asarray(step.prior)
    step.beta = 1.0

    annealer = _Annealer()
    accepted, invalid, rejected = sampler.walk_chains(annealer=annealer, step=step)

    assert invalid == 0
    assert accepted > 0  # a real, working M-H correction produced at least some moves

    # update(): scaling is delegated to the stepsizer component; no crash, sane bounds
    statistics = (accepted, invalid, rejected)
    assert sampler.update(annealer=annealer, statistics=statistics) is None
    assert sampler.stepsizer.min_step_size <= sampler.scaling <= sampler.stepsizer.max_step_size

    # all done
    return


# main
if __name__ == "__main__":
    test()


# end of file
