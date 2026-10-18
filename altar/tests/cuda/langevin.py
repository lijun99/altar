#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#


"""
Sanity check: altar.cuda.libcudaaltar.langevin's bindings, against a real GPU.

Covers {updateThetaBatched} (updates every column at once, with independent alpha1/alpha2
weights on the prior/data gradients).
"""


def test():
    import numpy
    import altar.cuda
    import altar

    langevin = altar.cuda.libcudaaltar.langevin

    rng = numpy.random.default_rng(9)
    samples, parameters = 10000, 5
    half_epsilon = 0.05

    # the batched update: every column at once, independent alpha1/alpha2 weights
    theta = altar.cuda.managed(shape=(samples, parameters), cell="float64")
    theta0 = rng.normal(size=(samples, parameters))
    numpy.asarray(theta)[:, :] = theta0

    prior_gradient = altar.cuda.managed(shape=(samples, parameters), cell="float64")
    numpy.asarray(prior_gradient)[:, :] = rng.normal(size=(samples, parameters))
    data_gradient = altar.cuda.managed(shape=(samples, parameters), cell="float64")
    numpy.asarray(data_gradient)[:, :] = rng.normal(size=(samples, parameters))
    eta = altar.cuda.managed(shape=(samples, parameters), cell="float64")
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
