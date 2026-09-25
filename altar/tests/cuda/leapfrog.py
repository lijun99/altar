#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# (c) 2013-2025 parasim inc
# (c) 2010-2025 california institute of technology
# all rights reserved


"""
Sanity check: altar.cuda.libcudaaltar.leapfrog's bindings, against a real GPU.

Covers the full Hamiltonian Monte Carlo leapfrog integrator: {sampleMomentum} (mean ~0, std
~1), {kineticEnergy} (0.5 * ||momentum||^2), {computePotentialAndGradient} (both without and
with a jacobian, the reparameterized path), {updatePosition}/{updateMomentum} (the leapfrog
position/momentum updates), {metropolis} (accept/reject test -- deltaH <= 0 always accepts),
and {restoreMatrix}/{restoreRejected} (undoing rejected proposals, row by row).
"""


def test():
    import numpy
    import pyre.grid
    import altar
    import altar.cuda

    leapfrog = altar.cuda.libcudaaltar.leapfrog

    rng = numpy.random.default_rng(4)
    samples, parameters = 20000, 4

    momentum = pyre.grid.managed(shape=(samples, parameters), cell="float64")
    numpy.asarray(momentum)[:, :] = 0.0
    leapfrog.cudaLeapfrog_sampleMomentum(momentum)
    m = numpy.asarray(momentum)
    assert abs(m.mean()) < 0.05 and abs(m.std() - 1.0) < 0.05

    kinetic = pyre.grid.managed(shape=(samples,), cell="float64")
    leapfrog.cudaLeapfrog_kineticEnergy(momentum, kinetic)
    assert numpy.allclose(numpy.asarray(kinetic), 0.5 * (m**2).sum(axis=1))

    prior = pyre.grid.managed(shape=(samples,), cell="float64")
    numpy.asarray(prior)[:] = rng.normal(size=samples)
    data = pyre.grid.managed(shape=(samples,), cell="float64")
    numpy.asarray(data)[:] = rng.normal(size=samples)
    grad_prior = pyre.grid.managed(shape=(samples, parameters), cell="float64")
    numpy.asarray(grad_prior)[:, :] = rng.normal(size=(samples, parameters))
    grad_data = pyre.grid.managed(shape=(samples, parameters), cell="float64")
    numpy.asarray(grad_data)[:, :] = rng.normal(size=(samples, parameters))
    potential = pyre.grid.managed(shape=(samples,), cell="float64")
    grad_potential = pyre.grid.managed(shape=(samples, parameters), cell="float64")
    beta = 0.7

    leapfrog.cudaLeapfrog_computePotentialAndGradient(
        prior, data, grad_prior, grad_data, potential, grad_potential, beta
    )
    pr, da = numpy.asarray(prior), numpy.asarray(data)
    gp, gd = numpy.asarray(grad_prior), numpy.asarray(grad_data)
    assert numpy.allclose(numpy.asarray(potential), -(pr + beta * da))
    assert numpy.allclose(numpy.asarray(grad_potential), -(gp + beta * gd))

    # the reparameterized path: the data gradient scaled elementwise by a jacobian
    jacobian = pyre.grid.managed(shape=(samples, parameters), cell="float64")
    numpy.asarray(jacobian)[:, :] = rng.uniform(0.5, 2.0, size=(samples, parameters))
    grad_potential_reparam = pyre.grid.managed(shape=(samples, parameters), cell="float64")
    leapfrog.cudaLeapfrog_computePotentialAndGradient(
        prior, data, grad_prior, grad_data, potential, grad_potential_reparam, beta, jacobian
    )
    jj = numpy.asarray(jacobian)
    assert numpy.allclose(numpy.asarray(grad_potential_reparam), -(gp + beta * jj * gd))

    theta = pyre.grid.managed(shape=(samples, parameters), cell="float64")
    numpy.asarray(theta)[:, :] = 1.0
    step = 0.01
    leapfrog.cudaLeapfrog_updatePosition(theta, momentum, step)
    assert numpy.allclose(numpy.asarray(theta), 1.0 + step * m)

    momentum2 = pyre.grid.managed(shape=(samples, parameters), cell="float64")
    numpy.asarray(momentum2)[:, :] = m
    scale = 0.5
    leapfrog.cudaLeapfrog_updateMomentum(momentum2, grad_potential, scale)
    assert numpy.allclose(numpy.asarray(momentum2), m + scale * numpy.asarray(grad_potential))

    deltaH = pyre.grid.managed(shape=(samples,), cell="float64")
    numpy.asarray(deltaH)[:] = rng.uniform(-3, 3, size=samples)
    mask = pyre.grid.managed(shape=(samples,), cell="int32")
    leapfrog.cudaLeapfrog_metropolis(deltaH, mask)
    mk = numpy.asarray(mask)
    dH = numpy.asarray(deltaH)
    # deltaH <= 0 always accepts (log(u) < 0 <= -deltaH for any u in (0,1))
    assert numpy.all(mk[dH <= 0] == 1)

    current = pyre.grid.managed(shape=(samples, parameters), cell="float64")
    numpy.asarray(current)[:, :] = 1.0
    backup = pyre.grid.managed(shape=(samples, parameters), cell="float64")
    numpy.asarray(backup)[:, :] = 2.0
    leapfrog.cudaLeapfrog_restoreMatrix(current, backup, mask)
    cu = numpy.asarray(current)
    assert numpy.all(cu[mk == 0] == 2.0)
    assert numpy.all(cu[mk == 1] == 1.0)

    theta_old = pyre.grid.managed(shape=(samples, parameters), cell="float64")
    numpy.asarray(theta_old)[:, :] = 5.0
    momentum_old = pyre.grid.managed(shape=(samples, parameters), cell="float64")
    numpy.asarray(momentum_old)[:, :] = 6.0
    theta2 = pyre.grid.managed(shape=(samples, parameters), cell="float64")
    numpy.asarray(theta2)[:, :] = 1.0
    momentum3 = pyre.grid.managed(shape=(samples, parameters), cell="float64")
    numpy.asarray(momentum3)[:, :] = 1.0
    leapfrog.cudaLeapfrog_restoreRejected(theta2, theta_old, momentum3, momentum_old, mask)
    th2, mo3 = numpy.asarray(theta2), numpy.asarray(momentum3)
    assert numpy.all(th2[mk == 0] == 5.0) and numpy.all(mo3[mk == 0] == 6.0)
    assert numpy.all(th2[mk == 1] == 1.0) and numpy.all(mo3[mk == 1] == 1.0)

    # all done
    return


# main
if __name__ == "__main__":
    test()


# end of file
