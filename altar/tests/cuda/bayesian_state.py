#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#


"""
Sanity check: {altar.bayesian.states.cuda.BayesianState}/{HMCState}, against a real GPU -- these are
the actual state classes the bayesian sampler layer allocates its buffers with, exercised here
completely unchanged, to confirm the {altar.cuda.matrix}/{altar.cuda.vector} alias
(altar/cuda/array.py) is a genuine drop-in for the old capsule-based buffers those classes
were written against.

Covers {start}/{alloc} (buffer allocation), {compute_posterior} (posterior = prior +
beta*data, via {altar.cuda.cublas.axpy}), {clone} (independent memory), and
{copy_from_cpu}/{copy_to_cpu} (interop with a cpu-side step, whose {altar.matrix}/
{altar.vector} buffers support the buffer protocol directly).
"""


class _FakeJob:
    def __init__(self, samples):
        self.chains = samples
        self.gpuprecision = "float64"


class _FakeModel:
    def __init__(self, samples, parameters, reparameterization=False):
        self.job = _FakeJob(samples)
        self.parameters = parameters
        self.reparameterization = reparameterization


class _FakeController:
    def __init__(self, model):
        self.model = model


def _check_bayesian_state():
    import numpy
    import altar
    from altar.bayesian.states.cuda.BayesianState import BayesianState

    samples, parameters = 1000, 4
    controller = _FakeController(_FakeModel(samples, parameters))

    step = BayesianState.start(controller)
    assert step.samples == samples and step.parameters == parameters
    assert numpy.all(numpy.asarray(step.theta) == 0.0)
    assert numpy.all(numpy.asarray(step.prior) == 0.0)

    rng = numpy.random.default_rng(2)
    numpy.asarray(step.theta)[:, :] = rng.normal(size=(samples, parameters))
    numpy.asarray(step.prior)[:] = rng.normal(size=samples)
    numpy.asarray(step.data)[:] = rng.normal(size=samples)
    step.beta = 0.5

    step.compute_posterior()
    expected = numpy.asarray(step.prior) + 0.5 * numpy.asarray(step.data)
    assert numpy.allclose(numpy.asarray(step.posterior), expected)

    clone = step.clone()
    assert numpy.array_equal(numpy.asarray(clone.theta), numpy.asarray(step.theta))
    numpy.asarray(clone.theta)[:, :] = 0.0
    assert not numpy.array_equal(numpy.asarray(clone.theta), numpy.asarray(step.theta))

    cpu_step = type("CPUStep", (), {})()
    cpu_step.beta = 0.75
    cpu_step.theta = altar.matrix(shape=(samples, parameters))
    numpy.asarray(cpu_step.theta)[:, :] = 42.0
    cpu_step.prior = altar.vector(shape=samples)
    numpy.asarray(cpu_step.prior)[:] = 7.0
    cpu_step.data = altar.vector(shape=samples)
    cpu_step.posterior = altar.vector(shape=samples)

    gpu_step = BayesianState.alloc(samples=samples, parameters=parameters, dtype="float64")
    gpu_step.copy_from_cpu(cpu_step)
    assert gpu_step.beta == 0.75
    assert numpy.all(numpy.asarray(gpu_step.theta) == 42.0)
    assert numpy.all(numpy.asarray(gpu_step.prior) == 7.0)

    cpu_out = type("CPUStep", (), {})()
    cpu_out.theta = altar.matrix(shape=(samples, parameters))
    cpu_out.prior = altar.vector(shape=samples)
    cpu_out.data = altar.vector(shape=samples)
    cpu_out.posterior = altar.vector(shape=samples)
    gpu_step.copy_to_cpu(cpu_out)
    assert cpu_out.beta == 0.75
    assert numpy.all(numpy.asarray(cpu_out.theta) == 42.0)

    # all done
    return


def _check_hmc_state():
    import numpy
    from altar.bayesian.states.cuda.HMCState import HMCState

    samples, parameters = 500, 3

    # the reparameterized path: a Jacobian buffer, initialized to 1
    controller = _FakeController(_FakeModel(samples, parameters, reparameterization=True))
    state = HMCState.start(controller)
    assert state.samples == samples and state.parameters == parameters
    assert state.reparameterization is True
    assert numpy.all(numpy.asarray(state.Jacobian) == 1.0)

    numpy.asarray(state.prior)[:] = 1.0
    numpy.asarray(state.data)[:] = 2.0
    state.beta = 0.5
    state.compute_posterior()
    assert numpy.allclose(numpy.asarray(state.posterior), 1.0 + 0.5 * 2.0)

    clone = state.clone()
    assert numpy.array_equal(numpy.asarray(clone.theta), numpy.asarray(state.theta))
    assert numpy.array_equal(numpy.asarray(clone.Jacobian), numpy.asarray(state.Jacobian))

    # the plain path: no Jacobian, phi is theta itself
    controller = _FakeController(_FakeModel(samples, parameters, reparameterization=False))
    state = HMCState.start(controller)
    assert state.reparameterization is False
    assert state.Jacobian is None
    assert state.phi is state.theta

    # all done
    return


def test():
    _check_bayesian_state()
    _check_hmc_state()
    return


# main
if __name__ == "__main__":
    test()


# end of file
