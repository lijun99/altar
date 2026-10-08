# -*- python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# externals
from __future__ import annotations
import math
import typing
# my base class
from .Base import Base


# declaration
class Brent(Base, family="altar.bayesian.solvers.brent"):
    """
    A δβ solver that finds the root of COV(δβ) - target with Brent's method; the COV of the
    weights increases with δβ, so the target is bracketed by 0, where the COV is 0, and the
    largest δβ, where it overshoots
    """


    # implementation details
    def dbeta(self, high: float) -> float:
        """
        The δβ in [0, {high}] whose COV is my target
        """
        target = self.target
        f = lambda x: self.cov_at(dbeta=x) - target
        return brent(f=f, a=0.0, b=high, fa=-target, fb=f(high), ftol=self.tolerance,
                     maxiter=self.maxiter)


EPS: float = 2.0 ** -52


def brent(f: typing.Callable[[float], float], a: float, b: float, fa: float, fb: float,
          ftol: float, xtol: float = 0.0, maxiter: int = 1000) -> float:
    """
    The root of {f} bracketed by [a, b], fa * fb < 0, by Brent's method: inverse quadratic
    interpolation or the secant when they converge fast enough, bisection otherwise; stop when
    |f| < {ftol}, the bracket is below {xtol} plus roundoff, or after {maxiter} evaluations
    """
    c, fc = b, fb
    d = e = b - a
    for _ in range(maxiter):
        # keep the root between b and c
        if (fb > 0) == (fc > 0):
            c, fc = a, fa
            d = e = b - a
        # b is the best guess so far
        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        tol = 2 * EPS * abs(b) + 0.5 * xtol
        m = 0.5 * (c - b)
        if abs(fb) < ftol or abs(m) <= tol or fb == 0:
            return b
        # try inverse quadratic interpolation, or the secant
        if abs(e) >= tol and abs(fa) > abs(fb):
            s = fb / fa
            if a == c:
                p, q = 2 * m * s, 1 - s
            else:
                q, r = fa / fc, fb / fc
                p = s * (2 * m * q * (q - r) - (b - a) * (r - 1))
                q = (q - 1) * (r - 1) * (s - 1)
            if p > 0:
                q = -q
            else:
                p = -p
            # accept it if it stays well within the bracket and shrinks fast enough
            if 2 * p < min(3 * m * q - abs(tol * q), abs(e * q)):
                e, d = d, p / q
            else:
                d = e = m
        # otherwise bisect
        else:
            d = e = m
        a, fa = b, fb
        b += d if abs(d) > tol else math.copysign(tol, m)
        fb = f(b)
    return b


# end of file
