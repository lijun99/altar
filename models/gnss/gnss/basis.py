# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# the basis functions of the trajectory model of a GNSS position time series, on times {t} in
# days since a reference epoch; each returns the columns of the design matrix, (len(t) x
# columns), with the same definitions as DISSTANS (Köhne et al., 2023), so that a DISSTANS fit
# and an AlTar model of the same series share their parameters:
#   polynomial   ((t - t0) / unit)^k, k = first..order
#   sinusoid     cos(2 pi (t - t0) / period), sin(2 pi (t - t0) / period)
#   step         H(t - ts)
#   logarithmic  log(1 + (t - ts) / tau) after ts, 0 before
#   ispline      the integral of the cardinal B-spline of {degree} and {scale} centered at c,
#                0 before c - (degree + 1) scale / 2 and 1 after c + (degree + 1) scale / 2
#   isplines     ispline columns of several scales with centers {spacing} days apart, inset
#                from the ends of a window by half their support (sse_disstans' DenseISplineSet;
#                DISSTANS' own SplineSet places them one scale apart), optionally scaled by
#                sqrt(scale / smallest scale) as DISSTANS' internal scaling does


# externals
from math import comb, factorial
import numpy


def polynomial(t, order=1, t0=0.0, unit=365.25, first=0):
    """
    The powers {first}..{order} of the time since {t0} in units of {unit} days
    """
    x = (numpy.asarray(t, dtype=float) - t0) / unit
    return x[:, None] ** numpy.arange(first, order + 1)[None, :]


def sinusoid(t, period=365.25, t0=0.0):
    """
    The cosine and sine of a {period} (days), with phase zero at {t0}
    """
    phase = 2 * numpy.pi * (numpy.asarray(t, dtype=float) - t0) / period
    return numpy.column_stack([numpy.cos(phase), numpy.sin(phase)])


def step(t, times):
    """
    A Heaviside step at each of {times}; NaN times give zero columns, for padding
    """
    t = numpy.asarray(t, dtype=float)[:, None]
    ts = numpy.atleast_1d(numpy.asarray(times, dtype=float))[None, :]
    with numpy.errstate(invalid="ignore"):
        return (t >= ts).astype(float)


def logarithmic(t, times, tau):
    """
    log(1 + (t - ts) / tau) after each of {times}, 0 before; NaN times give zero columns
    """
    t = numpy.asarray(t, dtype=float)[:, None]
    ts = numpy.atleast_1d(numpy.asarray(times, dtype=float))[None, :]
    tau = numpy.broadcast_to(numpy.asarray(tau, dtype=float), ts.shape)
    with numpy.errstate(invalid="ignore"):
        dt = t - ts
        out = numpy.where(dt >= 0, numpy.log1p(numpy.clip(dt, 0, None) / tau), 0.0)
    return numpy.nan_to_num(out)


def ispline(t, centers, scale, degree=2):
    """
    The I-splines of {degree} and {scale} (days) centered at each of {centers}
    """
    order = degree + 1
    x = (numpy.asarray(t, dtype=float)[:, None] - numpy.asarray(centers, dtype=float)[None, :]) / scale
    k = numpy.arange(order + 1)
    power = x[..., None] + order / 2 - k
    terms = (-1.0) ** k * numpy.array([comb(order, int(j)) for j in k]) \
        * numpy.where(power >= 0, power, 0.0) ** order
    out = terms.sum(axis=-1) / factorial(order)
    out[x < -order / 2] = 0.0
    out[x > order / 2] = 1.0
    return out


def ispline_centers(t_start, t_end, scale, spacing=1.0, degree=2):
    """
    The centers, {spacing} days apart, of the I-splines of {scale} whose ramps lie within
    [t_start, t_end]
    """
    half = scale * (degree + 1) / 2
    count = int((t_end - t_start - 2 * half) / spacing) + 1
    return t_start + half + spacing * numpy.arange(max(count, 0))


def isplines(t, t_start, t_end, scales, spacing=1.0, degree=2, internal_scaling=False):
    """
    The I-splines of each of {scales}, {spacing} days apart within [t_start, t_end]; returns
    the columns and a table of the scale and the center of each
    """
    columns, scale_of, center_of = [], [], []
    smallest = min(scales)
    for scale in scales:
        centers = ispline_centers(t_start, t_end, scale, spacing, degree)
        if not centers.size:
            continue
        g = ispline(t, centers, scale, degree)
        if internal_scaling:
            g = g * numpy.sqrt(scale / smallest)
        columns.append(g)
        scale_of.append(numpy.full(centers.size, float(scale)))
        center_of.append(centers)
    if not columns:
        return numpy.zeros((len(t), 0)), {"scale": numpy.zeros(0), "center": numpy.zeros(0)}
    return (numpy.hstack(columns),
            {"scale": numpy.concatenate(scale_of), "center": numpy.concatenate(center_of)})


# end of file
