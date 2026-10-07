#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#


"""
Check the mass matrices of {altar.bayesian.samplers.cuda.HMC} against a real GPU: the dense
products against numpy, and HMC on a strongly correlated gaussian, where a step size the dense
mass takes in stride leaves a diagonal mass rejecting almost every trajectory
"""


def test():
    import numpy
    import altar
    import altar.cuda
    from altar.bayesian.samplers.cuda.HMC import HMC
    from altar.bayesian.stepsizers.StepSizer import FixedStepSize

    samples = 2000
    parameters = 3
    # the target: N(0, Σ), with sd (1, 2, 0.5) and a 0.995 correlation of the first two
    sd = numpy.array([1.0, 2.0, 0.5])
    r = numpy.eye(parameters)
    r[0, 1] = r[1, 0] = 0.995
    Σ = r * numpy.outer(sd, sd)
    precision = numpy.linalg.inv(Σ)

    class _Model:
        reparameterization = False

        def likelihoods(self, annealer, step):
            θ = numpy.asarray(step.theta)
            numpy.asarray(step.prior)[:] = -0.5 * numpy.einsum("ij,jk,ik->i", θ, precision, θ)
            numpy.asarray(step.data)[:] = 0.0

        def gradient(self, controller, step, batch=None):
            numpy.asarray(step.prior_gradient)[:, :] = -numpy.asarray(step.theta) @ precision
            numpy.asarray(step.data_gradient)[:, :] = 0.0

    _Model.parameters = parameters

    class _Dispatcher:
        sample_posterior_start = sample_posterior_finish = None
        chain_advance_start = chain_advance_finish = None

        def notify(self, event, controller):
            return

    class _Job:
        chains = samples
        gpuprecision = "float64"
        steps = 5

    class _Info:
        @staticmethod
        def log(*args, **kwds):
            return

    class _Application:
        job = _Job()
        info = _Info()

    class _Annealer:
        model = _Model()
        dispatcher = _Dispatcher()

    def matrix(source):
        m = altar.matrix(shape=source.shape)
        numpy.asarray(m)[:, :] = source
        return m

    def vector(source):
        v = altar.vector(shape=source.shape[0])
        numpy.asarray(v)[:] = source
        return v

    class _Step:
        # the chains start from the target, which is also the population the mass comes from
        def __init__(self, θ):
            self.beta = 1.0
            self.theta = matrix(θ)
            self.weighted_theta = matrix(θ)
            self.weights = vector(numpy.full(samples, 1 / samples))
            self.momentum = None
            for name in ("prior", "data", "posterior", "U", "H"):
                setattr(self, name, altar.vector(shape=samples))
            for name in ("prior_gradient", "data_gradient", "U_gradient"):
                setattr(self, name, altar.matrix(shape=(samples, parameters)))

    def sampler(mass):
        hmc = HMC()
        hmc.stepsizer = FixedStepSize()
        hmc.mass_matrix = mass
        hmc.mass_shrinkage = 0.0
        hmc.initialize(application=_Application())
        hmc._allocate(model=_Model())
        hmc.leapfrog_steps = 10
        hmc.step_size = 0.5
        return hmc

    rng = numpy.random.default_rng(0)
    draw = lambda: rng.multivariate_normal(numpy.zeros(parameters), Σ, size=samples)
    annealer = _Annealer()

    # the dense mass: the momenta, the velocity and the kinetic energy against numpy
    dense = sampler("dense")
    dense._set_mass(_Step(draw()))
    assert dense._dense
    P = rng.normal(size=(samples, parameters))
    numpy.asarray(dense.proposal_state.momentum)[:, :] = P
    M = numpy.asarray(dense._inverse_mass)
    L = numpy.asarray(dense._factor)
    assert numpy.allclose(L @ L.T, M)
    assert numpy.allclose(numpy.asarray(dense._scale(dense._inverse_mass)), P @ M)
    assert numpy.allclose(numpy.asarray(dense._scale(dense._inverse_factor)), P @ numpy.linalg.inv(L))
    energy = altar.cuda.vector(shape=samples, dtype="float64").zero()
    dense._kinetic(energy)
    assert numpy.allclose(numpy.asarray(energy), 0.5 * numpy.einsum("ij,jk,ik->i", P, M, P))

    # HMC with the dense mass: high acceptance, and the population keeps the target
    step = _Step(draw())
    accepted = attempts = 0
    for _ in range(20):
        stats = dense.sample_posterior(annealer=annealer, step=step)
        accepted += stats.accepted
        attempts += stats.accepted + stats.rejected
    assert accepted / attempts > 0.8
    θ = numpy.asarray(step.theta)
    assert numpy.allclose(θ.std(0), sd, rtol=0.1)
    assert abs(numpy.corrcoef(θ.T)[0, 1] - 0.995) < 0.005

    # the same step with a diagonal mass is far too long for the narrow direction
    diagonal = sampler("diagonal")
    stats = diagonal.sample_posterior(annealer=annealer, step=_Step(draw()))
    assert not diagonal._dense
    assert stats.accepted / (stats.accepted + stats.rejected) < 0.2

    # too few chains per parameter: a diagonal mass, however many rows the population has
    few = sampler("dense")
    few.dense_mass_chains = 1000
    few._set_mass(_Step(draw()))
    assert not few._dense
    assert few._factor.shape == few.proposal_state.momentum.shape

    # all done
    return


# main
if __name__ == "__main__":
    test()


# end of file
