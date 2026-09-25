#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#


"""
Sanity check: {altar.bayesian.samplers.cuda.MetropolisVaryingSteps}, against a real GPU.

Same bug class as {Metropolis} (see {metropolis_sampler.py}): {resample}/
{@altar.provides} -> {update}/{@altar.export}, {libcudaaltar.cudaMetropolis_*}/{.data} ->
{libcudaaltar.metropolis.cudaMetropolis_*}/{.grid}, {.Cholesky(uplo=...)} -> {.cholesky()},
keyword-style {cublas.trmm(...)} -> the real {cublas.dtrmm}/{strmm} positional call, and
{curand.uniform(self.curng, out=dice)}'s argument order fixed. This class additionally had a
dead reference to a module that never existed anywhere in this codebase,
{altar.cuda.stats.correlation(...)} -- its {walk_chains} loop runs until the population's
per-parameter correlation with its starting state drops below {target_correlation}, so that
check is now computed directly off the (host-visible, managed-memory) numpy views instead.

{target_correlation}/{corr_check_steps}/{max_mc_steps} are all overridden to small values here
so the loop exits after one or two correlation checks rather than running the real defaults
(target 0.6, checked every 1000 steps, up to 100000 steps) -- this test only needs to prove
the loop runs correctly end-to-end on real GPU data, not reach statistical equilibrium (that
full-equilibration check already lives in {metropolis_sampler.py}, which exercises the shared
{walk_chains}/{displace} logic against a standard-normal target).
"""


def test():
    import numpy
    import altar
    from altar.bayesian.samplers.cuda.MetropolisVaryingSteps import MetropolisVaryingSteps

    samples, parameters = 500, 3

    class _Dispatcher:
        def __getattr__(self, name):
            return name

        def notify(self, event, controller):
            return

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

    sampler = MetropolisVaryingSteps(name="cuda-metropolis-varying-steps-test")
    sampler.curng = altar.cuda.curand_generator()
    sampler.precision = "float64"
    sampler.scaling = 0.5
    # exit quickly: a handful of correlation checks, not the real defaults
    sampler.corr_check_steps = 5
    sampler.max_mc_steps = 20
    sampler.target_correlation = 0.999

    sampler.allocate_gpu_data(samples, parameters)
    numpy.asarray(sampler.gsigma_chol)[:, :] = numpy.eye(parameters)
    sampler.gsigma_chol.cholesky()
    sampler.gsigma_chol *= sampler.scaling

    step = sampler.CoolingStep.alloc(samples, parameters, dtype="float64")
    rng = numpy.random.default_rng(2)
    numpy.asarray(step.theta)[:, :] = rng.normal(size=(samples, parameters))
    numpy.asarray(step.prior)[:] = -0.5 * numpy.sum(numpy.asarray(step.theta) ** 2, axis=1)
    numpy.asarray(step.data)[:] = 0.0
    numpy.asarray(step.posterior)[:] = numpy.asarray(step.prior)
    step.beta = 1.0

    annealer = _Annealer()
    accepted, invalid, rejected = sampler.walk_chains(annealer=annealer, step=step)

    attempts = accepted + rejected + invalid
    assert attempts == sampler.corr_check_steps * samples or attempts == sampler.max_mc_steps * samples
    assert invalid == 0
    assert accepted > 0  # a real, working M-H correction produced at least some moves

    # update()/adjust_covariance_scaling: no crash, scaling stays in a sane range
    statistics = (accepted, invalid, rejected)
    assert sampler.update(annealer=annealer, statistics=statistics) is None
    assert 0.1 <= sampler.scaling <= 1.0

    # all done
    return


# main
if __name__ == "__main__":
    test()


# end of file
