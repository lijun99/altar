#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#


"""
Sanity check: the {altar.distributions.cuda} implementation classes (Gaussian, Uniform,
TGaussian), against a real GPU -- one level above the raw {libcudaaltar.distributions}
bindings altar/tests/cuda/{gaussian,uniform,tgaussian}.py already cover directly.

This exercises the actual python classes the framework picks at runtime
({altar.distributions.cuda.Gaussian.Gaussian} etc.), not the pybind11 functions directly, to
catch calling-convention bugs at that layer (argument order, tuple-vs-separate-args, stale
{.data} attribute access, wrong module path) that the lower-level tests can't see.

Built and torn down without pyre's component machinery: a pre-existing pyre bug blocks live
instantiation of any pyre component with an {altar.properties.array} trait (which the real
shims -- {altar.distributions.Uniform}, {TGaussian} -- both have, for their {support} trait),
so this instantiates the plain backend classes directly and sets their configuration
attributes by hand, exactly as the shim's {_makeImpl} would.
"""


class _FakeApplication:
    class controller:
        class worker:
            device = "gpu0"

    class job:
        gpuprecision = "float64"


def test():
    import numpy
    import altar.cuda

    from altar.distributions.cuda.Gaussian import Gaussian
    from altar.distributions.cuda.Uniform import Uniform
    from altar.distributions.cuda.TGaussian import TGaussian

    samples, parameters = 20000, 4
    application = _FakeApplication()

    # Gaussian, occupying columns [1, 3)
    gaussian = Gaussian()
    gaussian.offset, gaussian.parameters = 1, 2
    gaussian.mean, gaussian.sigma = 2.0, 1.5
    gaussian.initialize(rng=None, application=application)
    assert (gaussian.idx_begin, gaussian.idx_end) == (1, 3)

    theta = altar.cuda.managed(shape=(samples, parameters), cell="float64")
    numpy.asarray(theta)[:, :] = 0.0
    gaussian.initialize_sample(theta)
    arr = numpy.asarray(theta)
    assert numpy.all(arr[:, 0] == 0.0) and numpy.all(arr[:, 3] == 0.0)
    sub = arr[:, 1:3]
    assert abs(sub.mean() - 2.0) < 0.05 and abs(sub.std() - 1.5) < 0.05

    likelihood = altar.cuda.managed(shape=(samples,), cell="float64")
    numpy.asarray(likelihood)[:] = 0.0
    gaussian.eval_prior(theta, likelihood)
    expected = numpy.zeros(samples)
    c1 = -numpy.log(1.5 * numpy.sqrt(2 * numpy.pi))
    c2 = 0.5 / 1.5**2
    for i in (1, 2):
        expected += c1 - c2 * (arr[:, i] - 2.0) ** 2
    assert numpy.allclose(numpy.asarray(likelihood), expected, atol=1e-8)

    gradient = altar.cuda.managed(shape=(samples, parameters), cell="float64")
    numpy.asarray(gradient)[:, :] = -999.0
    gaussian.prior_gradient(theta, gradient)
    gr = numpy.asarray(gradient)
    for i in (1, 2):
        assert numpy.allclose(gr[:, i], (2.0 - arr[:, i]) / 1.5**2)
    assert numpy.all(gr[:, 0] == -999.0) and numpy.all(gr[:, 3] == -999.0)

    # Uniform, occupying columns [0, 2)
    uniform = Uniform()
    uniform.offset, uniform.parameters = 0, 2
    uniform.support = (-2.0, 5.0)
    uniform.initialize(rng=None, application=application)

    theta = altar.cuda.managed(shape=(samples, parameters), cell="float64")
    numpy.asarray(theta)[:, :] = 0.0
    uniform.initialize_sample(theta)
    arr = numpy.asarray(theta)
    assert numpy.all(arr[:, 2] == 0.0) and numpy.all(arr[:, 3] == 0.0)
    sub = arr[:, 0:2]
    assert sub.min() >= -2.0 and sub.max() < 5.0

    likelihood = altar.cuda.managed(shape=(samples,), cell="float64")
    numpy.asarray(likelihood)[:] = 0.0
    uniform.eval_prior(theta, likelihood)
    assert numpy.allclose(numpy.asarray(likelihood), -numpy.log(7.0) * 2)

    mask = altar.cuda.managed(shape=(samples,), cell="int32")
    numpy.asarray(mask)[:] = 0
    arr[0, 0] = 100.0  # push one sample out of range
    uniform.verify(theta, mask)
    assert numpy.asarray(mask)[0] == 1

    clamped = altar.cuda.managed(shape=(samples, parameters), cell="float64")
    numpy.asarray(clamped)[:, :] = 100.0
    uniform.constrain(clamped)
    cc = numpy.asarray(clamped)
    assert numpy.all(cc[:, 0] == 5.0) and numpy.all(cc[:, 1] == 5.0)  # clamped to high
    assert numpy.all(cc[:, 2] == 100.0)  # untouched, outside [idx_begin, idx_end)

    # TGaussian, occupying columns [1, 3)
    tgaussian = TGaussian()
    tgaussian.offset, tgaussian.parameters = 1, 2
    tgaussian.mean, tgaussian.sigma = 1.0, 2.0
    tgaussian.support = (0.1, 0.9)
    tgaussian.initialize(rng=None, application=application)

    theta = altar.cuda.managed(shape=(samples, parameters), cell="float64")
    numpy.asarray(theta)[:, :] = 0.5  # within the raw support
    mask = altar.cuda.managed(shape=(samples,), cell="int32")
    numpy.asarray(mask)[:] = 0
    tgaussian.verify(theta, mask)
    assert numpy.all(numpy.asarray(mask) == 0)

    arr = numpy.asarray(theta)
    arr[0, 1] = 1000.0  # push one sample's parameter 1 out of the raw support
    mask = altar.cuda.managed(shape=(samples,), cell="int32")
    numpy.asarray(mask)[:] = 0
    tgaussian.verify(theta, mask)
    mk = numpy.asarray(mask)
    assert mk[0] == 1 and numpy.all(mk[1:] == 0)

    likelihood = altar.cuda.managed(shape=(samples,), cell="float64")
    numpy.asarray(likelihood)[:] = 0.0
    tgaussian.eval_prior(theta, likelihood)  # just needs to run without error

    gradient = altar.cuda.managed(shape=(samples, parameters), cell="float64")
    numpy.asarray(gradient)[:, :] = -999.0
    tgaussian.prior_gradient(theta, gradient)
    gr = numpy.asarray(gradient)
    for i in (1, 2):
        assert numpy.allclose(gr[:, i], (1.0 - arr[:, i]) / 2.0**2)

    # all done
    return


# main
if __name__ == "__main__":
    test()


# end of file
