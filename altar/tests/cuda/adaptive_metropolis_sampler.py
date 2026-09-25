#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#


"""
Sanity check: {altar.bayesian.samplers.cuda.AdaptiveMetropolis}, against a real GPU.

Same bug class as {Metropolis}/{MetropolisVaryingSteps} (see
{metropolis_sampler.py}/{metropolis_varying_steps_sampler.py}): {resample}/{@altar.provides}
-> {update}/{@altar.export}, {libcudaaltar.cudaMetropolis_*}/{.data} ->
{libcudaaltar.metropolis.cudaMetropolis_*}/{.grid}, {.Cholesky(uplo=...)} -> {.cholesky()},
keyword-style {cublas.trmm(...)} -> the real {cublas.dtrmm}/{strmm} positional call, the
{curand.uniform}/{gaussian} argument-order fix, and the dead
{altar.cuda.stats.correlation(...)} reference replaced with a direct numpy computation.

This class additionally has its own scaling-adaptation rule ({adjust_covariance_scaling}'s
Robbins-Monro-style feedback-gain update, `scaling *= exp(gain*(acceptance-target))`, unlike
the other two samplers' acceptance-weighted blend) and a `min_mc_steps`/`max_mc_steps`/
`beta_stage2` staged-length walk_chains loop -- neither touched by the bug fix, both exercised
here as-is (with the correlation thresholds relaxed so the test exits quickly).
"""


def test():
    import numpy
    import altar
    from altar.bayesian.samplers.cuda.AdaptiveMetropolis import AdaptiveMetropolis

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

    sampler = AdaptiveMetropolis(name="cuda-adaptive-metropolis-test")
    sampler.curng = altar.cuda.curand_generator()
    sampler.precision = "float64"
    sampler.info = _Info()
    sampler.scaling = 0.5
    sampler.scaling_min, sampler.scaling_max = 0.01, 1.0
    # {gain_function} needs scipy.special.erfcinv, not installed in this env; not part of
    # the api-mismatch bug this test targets, so just set a plausible gain directly
    sampler.gain = 2.1
    # exit quickly: relax the correlation-based stopping loop
    sampler.min_mc_steps = 5
    sampler.corr_check_steps = 5
    sampler.max_mc_steps = 10
    sampler.max_mc_steps_stage2 = 10
    sampler.target_correlation = 0.999

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

    # update()/adjust_covariance_scaling: no crash, scaling stays in a sane range
    statistics = (accepted, invalid, rejected)
    assert sampler.update(annealer=annealer, statistics=statistics) is None
    assert sampler.scaling_min <= sampler.scaling <= sampler.scaling_max

    # all done
    return


# main
if __name__ == "__main__":
    test()


# end of file
