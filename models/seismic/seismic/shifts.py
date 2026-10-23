# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# the time shift of each station's traces, by Keys cubic interpolation, on the cpu: the
# observations are the traces of the stations, {traces} per station and {length} samples each,
# station by station; {delays} are the shifts in samples, (samples x stations); a positive delay
# moves the traces later, y[i] = x(i - delay), with zeros beyond their ends; as the cuda kernels
# of libcudaseismic/cudaTimeShift
import numpy


def keys(x):
    """
    The Keys cubic kernel, a = -1/2
    """
    x = numpy.abs(x)
    return numpy.where(x < 1, (1.5 * x - 2.5) * x * x + 1, numpy.where(x < 2, ((-0.5 * x + 2.5) * x - 4) * x + 2, 0.0))


def keys_derivative(x):
    """
    The derivative of {keys}
    """
    sign, x = numpy.sign(x), numpy.abs(x)
    return sign * numpy.where(x < 1, (4.5 * x - 5) * x, numpy.where(x < 2, (-1.5 * x + 5) * x - 4, 0.0))


def _taps(x, positions, kernel):
    """
    sum over the four samples j of each trace around {positions} of x[j] kernel(positions - j)
    """
    length = x.shape[-1]
    j0 = numpy.floor(positions).astype(int)
    total = numpy.zeros(positions.shape)
    for k in range(-1, 3):
        j = j0 + k
        inside = (j >= 0) & (j < length)
        values = numpy.take_along_axis(x, numpy.clip(j, 0, length - 1), axis=-1)
        total += numpy.where(inside, values * kernel(positions - j), 0.0)
    return total


def shift(x, delays, traces, length):
    """
    The traces of {x}, (samples x observations), each station's shifted by its delay
    """
    rows = x.shape[0]
    y = x.reshape(rows, -1, traces, length)
    i = numpy.arange(length)
    positions = numpy.broadcast_to(i - delays[:, :, None, None], y.shape)
    return _taps(y, positions, keys).reshape(rows, -1)


def adjoint(w, delays, traces, length):
    """
    The transpose of {shift} applied to {w}: out[j] = sum_i w[i] keys(i - delay - j)
    """
    rows = w.shape[0]
    v = w.reshape(rows, -1, traces, length)
    j = numpy.arange(length)
    # the outputs i that read input j are around j + delay; keys is even
    positions = numpy.broadcast_to(j + delays[:, :, None, None], v.shape)
    return _taps(v, positions, keys).reshape(rows, -1)


def gradient(x, w, delays, traces, length):
    """
    The derivatives of w . shift(x) with respect to the delays, (samples x stations), in samples
    """
    rows = x.shape[0]
    y = x.reshape(rows, -1, traces, length)
    i = numpy.arange(length)
    positions = numpy.broadcast_to(i - delays[:, :, None, None], y.shape)
    # d y_i / d delay = -sum_j x_j keys'(i - delay - j)
    dy = -_taps(y, positions, keys_derivative)
    return (w.reshape(dy.shape) * dy).sum(axis=(2, 3))


# end of file
