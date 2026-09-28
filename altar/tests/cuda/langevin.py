#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#


"""
Sanity check: altar.cuda.libcudaaltar.langevin's bindings, against a real GPU.

Covers {updateTheta} (updates a single column of theta, leaves the others untouched) and
{updateThetaBatched} (updates every column at once, with independent alpha1/alpha2 weights on
the prior/data gradients).
"""


def test():
    import numpy
    import pyre.cuda
    import altar
    import altar.cuda

    langevin = altar.cuda.libcudaaltar.langevin

    rng = numpy.random.default_rng(9)
    samples, parameters = 10000, 5

    theta = pyre.cuda.managed(shape=(samples, parameters), cell="float64")
    theta0 = rng.normal(size=(samples, parameters))
    numpy.asarray(theta)[:, :] = theta0

    prior_gradient = pyre.cuda.managed(shape=(samples,), cell="float64")
    numpy.asarray(prior_gradient)[:] = rng.normal(size=samples)
    data_gradient = pyre.cuda.managed(shape=(samples,), cell="float64")
    numpy.asarray(data_gradient)[:] = rng.normal(size=samples)
    eta = pyre.cuda.managed(shape=(samples,), cell="float64")
    numpy.asarray(eta)[:] = rng.normal(size=samples) * 0.01
    half_epsilon = 0.05
    index = 2

    langevin.cudaLangevin_updateTheta(theta, prior_gradient, data_gradient, half_epsilon, eta, index)
    th = numpy.asarray(theta)
    pg, dg, et = numpy.asarray(prior_gradient), numpy.asarray(data_gradient), numpy.asarray(eta)
    expected = theta0[:, index] + half_epsilon * (pg + dg) + et
    assert numpy.allclose(th[:, index], expected)
    for c in range(parameters):
        if c != index:
            assert numpy.allclose(th[:, c], theta0[:, c])

    # the batched update: every column at once, independent alpha1/alpha2 weights
    theta = pyre.cuda.managed(shape=(samples, parameters), cell="float64")
    theta0 = rng.normal(size=(samples, parameters))
    numpy.asarray(theta)[:, :] = theta0

    prior_gradient = pyre.cuda.managed(shape=(samples, parameters), cell="float64")
    numpy.asarray(prior_gradient)[:, :] = rng.normal(size=(samples, parameters))
    data_gradient = pyre.cuda.managed(shape=(samples, parameters), cell="float64")
    numpy.asarray(data_gradient)[:, :] = rng.normal(size=(samples, parameters))
    eta = pyre.cuda.managed(shape=(samples, parameters), cell="float64")
    numpy.asarray(eta)[:, :] = rng.normal(size=(samples, parameters)) * 0.01
    alpha1, alpha2 = 1.0, 0.8

    langevin.cudaLangevin_updateThetaBatched(
        theta, alpha1, prior_gradient, alpha2, data_gradient, half_epsilon, eta
    )
    pg, dg, et = numpy.asarray(prior_gradient), numpy.asarray(data_gradient), numpy.asarray(eta)
    expected = theta0 + half_epsilon * (alpha1 * pg + alpha2 * dg) + et
    assert numpy.allclose(numpy.asarray(theta), expected)

    # all done
    return


# main
if __name__ == "__main__":
    test()


# end of file
