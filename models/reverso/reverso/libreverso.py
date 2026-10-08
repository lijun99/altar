# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis (michael.aivazis@para-sim.com)
# grace bato           (mary.grace.p.bato@jpl.nasa.gov)
# eric m. gurrola      (eric.m.gurrola@jpl.nasa.gov)
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved


# externals
import numpy


def gamma(sill, v):
    """
    The volume change of a chamber of radius a per unit overpressure, in units of pi a^3 / G
    """
    return 8 * (1 - v) / (3 * numpy.pi) if sill else 1.0


def response(sill, r, H, a, G, v):
    """
    The surface displacement, radial and up, at distance {r} per unit overpressure of a chamber
    of radius {a} at depth {H}
    """
    R2 = r**2 + H**2
    alpha = 4 * H**2 / (numpy.pi * R2) if sill else 1.0
    f = a**3 * alpha * (1 - v) / (G * R2**1.5)
    return r * f, H * f


def REVERSO(t, x, y, Qin, H_s, H_d, a_s, a_d, a_c, G, v, mu, drho, g,
            shallow_sill=True, deep_sill=True):
    """
    The (east, north, up) surface displacements at times {t} and locations ({x}, {y}) of the
    two magma chamber model of Reverso et al. [2014], starting from zero overpressures

    model parameters:
        Qin: the basal magma inflow rate
        H_s, a_s: the depth and radius of the shallow chamber
        H_d, a_d: the depth and radius of the deep chamber
        a_c: the radius of the conduit connecting them
    """
    t, x, y = (numpy.asarray(c, dtype=float) for c in (t, x, y))
    pi = numpy.pi
    gamma_s = gamma(shallow_sill, v)
    gamma_d = gamma(deep_sill, v)

    # the ratio of the chamber volumes and the length of the conduit
    k = (a_d/a_s)**3
    H_c = H_d - H_s
    gamma_r = gamma_s + gamma_d*k
    # the characteristic time (eq. 10)
    tau = 8 * mu * H_c * gamma_s * gamma_d * k * a_s**3 / (G * a_c**4 * gamma_r)
    # the amplitude of the transient
    A = gamma_d*k / gamma_r * (drho*g*H_c - 8*gamma_s*mu*Qin*H_c / (pi * a_c**4 * gamma_r))

    # the overpressures
    f0 = A * (1 - numpy.exp(-t/tau))
    f1 = G * Qin * t / (pi * a_s**3 * gamma_r)
    dP_s = f1 + f0
    dP_d = f1 - f0 * gamma_s / (gamma_d*k)

    # the displacements
    r = numpy.sqrt(x**2 + y**2)
    ur_s, uz_s = response(shallow_sill, r, H_s, a_s, G, v)
    ur_d, uz_d = response(deep_sill, r, H_d, a_d, G, v)
    ur = ur_s*dP_s + ur_d*dP_d
    uz = uz_s*dP_s + uz_d*dP_d
    phi = numpy.arctan2(y, x)
    return ur * numpy.cos(phi), ur * numpy.sin(phi), uz


# end of file
