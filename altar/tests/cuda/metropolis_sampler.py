#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# (c) 2013-2025 parasim inc
# (c) 2010-2025 california institute of technology
# all rights reserved


"""
Sanity check: {altar.bayesian.samplers.cuda.Metropolis}, against a real GPU.

Covers the full Metropolis sampler: protocol conformance (this class's {resample}/
{@altar.provides} was fixed to {update}/{@altar.export}, matching {Sampler}, and its
{libcudaaltar.cudaMetropolis_*}/{.data} calls were fixed to {libcudaaltar.metropolis.
cudaMetropolis_*}/{.grid}, the same class of bug {altar.bayesian.samplers.cuda.HMC} had),
{displace} (the Cholesky
+ triangular-matrix-multiply random walk step, checked by comparing 20000 draws' sample
covariance against the target covariance), and {walk_chains} (the complete accept/reject
loop, driven directly rather than through {sample_posterior}, since the latter's
{prepare_sampling_pdf} reaches into {annealer.worker.gstep} -- a real worker/model dependency
outside this test's scope, same as every other cuda/bayesian sampler).

The {walk_chains} target is a standard normal, same as {hmc_sampler.py}: starting near a
point mass, the population should equilibrate to std ~= 1 with a plausible (not 100%, not
nowhere-near-0%) acceptance rate. A first version of this test used a fake model whose
{likelihoods} didn't set {step.posterior} -- every candidate was then accepted regardless of
its actual posterior (a real model's {likelihoods} always computes prior+data+posterior
together, the same contract {altar.bayesian.samplers.Metropolis}/{HMC} rely on), and the
chain diverged into an uncorrected random walk instead of equilibrating; this is why the fake
model here explicitly computes {posterior = prior + beta*data}.
"""


def test():
    import numpy
    import altar
    from altar.bayesian.samplers.cuda.Metropolis import Metropolis

    samples, parameters = 2000, 3

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

    sampler = Metropolis(name="cuda-metropolis-test")
    sampler.curng = altar.cuda.curand_generator()
    sampler.precision = "float64"
    sampler.mcsteps = 1
    sampler.scaling = 0.5

    sampler.allocate_gpu_data(samples, parameters)
    numpy.asarray(sampler.gsigma_chol)[:, :] = numpy.eye(parameters)
    sampler.gsigma_chol.cholesky()
    sampler.gsigma_chol *= sampler.scaling

    # the covariance random walk itself: displace() over 20000 draws should reproduce the
    # target covariance statistically
    covariance_root = numpy.random.default_rng(3).random((parameters, parameters))
    covariance = covariance_root @ covariance_root.T + parameters * numpy.eye(parameters)
    check_chol = altar.cuda.matrix(shape=(parameters, parameters), dtype="float64")
    numpy.asarray(check_chol)[:, :] = covariance
    check_chol.cholesky()
    sampler_for_displace = Metropolis(name="cuda-metropolis-displace-test")
    sampler_for_displace.curng = altar.cuda.curand_generator()
    sampler_for_displace.gsigma_chol = check_chol
    displacement = altar.cuda.matrix(shape=(20000, parameters), dtype="float64").zero()
    sampler_for_displace.displace(displacement=displacement)
    sample_covariance = numpy.cov(numpy.asarray(displacement).T)
    assert numpy.allclose(sample_covariance, covariance, atol=0.2)

    # the full walk_chains loop
    step = sampler.CoolingStep.alloc(samples, parameters, dtype="float64")
    rng = numpy.random.default_rng(1)
    numpy.asarray(step.theta)[:, :] = rng.normal(size=(samples, parameters))
    numpy.asarray(step.prior)[:] = -0.5 * numpy.sum(numpy.asarray(step.theta) ** 2, axis=1)
    numpy.asarray(step.data)[:] = 0.0
    numpy.asarray(step.posterior)[:] = numpy.asarray(step.prior)
    step.beta = 1.0

    annealer = _Annealer()
    total_accepted = total_rejected = total_invalid = 0
    for _ in range(80):
        accepted, invalid, rejected = sampler.walk_chains(annealer=annealer, step=step)
        total_accepted += accepted
        total_rejected += rejected
        total_invalid += invalid

    arr = numpy.asarray(step.theta)
    assert total_invalid == 0
    assert abs(arr.mean()) < 0.1
    assert abs(arr.std() - 1.0) < 0.15
    attempts = total_accepted + total_rejected + total_invalid
    assert 0.3 < total_accepted / attempts < 0.95  # a real, working M-H correction, not 100%

    # update()/adjust_covariance_scaling
    statistics = (total_accepted, total_invalid, total_rejected)
    assert sampler.update(annealer=annealer, statistics=statistics) is None
    assert sampler.scalingMin <= sampler.scaling <= sampler.scalingMax

    # all done
    return


# main
if __name__ == "__main__":
    test()


# end of file
