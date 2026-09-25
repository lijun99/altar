#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#


"""
Sanity check: altar.cuda.libcudaaltar.distributions's cudaLogistic bindings, against a real
GPU.

Covers {sample} (mean ~0, std ~pi/sqrt(3), the standard logistic distribution's moments),
{logpdf} (an accumulation, checked against log_pdf(x) = x - 2*log(1+e^x)), and
{logpdfgradient} (checked against a central finite difference of that same log_pdf formula --
this specifically guards a real bug found while porting: the pre-port code computed
2*e^x - 1 for the gradient instead of the correct 2/(1+e^x) - 1).
"""


def test():
    import numpy
    import pyre.grid
    import altar
    import altar.cuda

    distributions = altar.cuda.libcudaaltar.distributions

    samples, parameters = 20000, 3
    idx_begin, idx_end = 0, 2

    theta = pyre.grid.managed(shape=(samples, parameters), cell="float64")
    numpy.asarray(theta)[:, :] = 0.0
    distributions.cudaLogistic_sample(theta, idx_begin, idx_end)
    arr = numpy.asarray(theta)
    assert numpy.all(arr[:, 2] == 0.0)

    sub = arr[:, idx_begin:idx_end]
    assert abs(sub.mean()) < 0.05
    assert abs(sub.std() - numpy.pi / numpy.sqrt(3)) < 0.05

    probability = pyre.grid.managed(shape=(samples,), cell="float64")
    numpy.asarray(probability)[:] = 0.0
    distributions.cudaLogistic_logpdf(theta, probability, idx_begin, idx_end)
    got = numpy.asarray(probability)
    expected = numpy.zeros(samples)
    for i in range(idx_begin, idx_end):
        x = arr[:, i]
        expected += x - 2 * numpy.log(1 + numpy.exp(x))
    assert numpy.allclose(got, expected, atol=1e-8)

    # the gradient, checked against a central finite difference of log_pdf itself
    gradient = pyre.grid.managed(shape=(samples, parameters), cell="float64")
    numpy.asarray(gradient)[:, :] = -999.0
    distributions.cudaLogistic_logpdfgradient(theta, gradient, idx_begin, idx_end)
    gp = numpy.asarray(gradient)

    def log_pdf(x):
        return x - 2 * numpy.log(1 + numpy.exp(x))

    h = 1e-5
    for i in range(idx_begin, idx_end):
        x = arr[:, i]
        finite_difference = (log_pdf(x + h) - log_pdf(x - h)) / (2 * h)
        assert numpy.allclose(gp[:, i], finite_difference, atol=1e-4)
    assert numpy.all(gp[:, 2] == -999.0)

    # all done
    return


# main
if __name__ == "__main__":
    test()


# end of file
