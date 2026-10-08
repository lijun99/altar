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
import numpy


# the declaration
class L2:
    """
    The cpu implementation of the L2 norm, of a vector {v} or of each row of a
    (samples x observations) {v}; {sigma_inv} is 1/sigma, or L, the lower Cholesky factor of
    the inverse covariance L L^T
    """


    def eval(self, v: numpy.ndarray, sigma_inv: float | numpy.ndarray | None = None,
             batch: int | None = None) -> float | numpy.ndarray:
        """
        The L2 norm of {v}, or of each of its first {batch} rows, with or without a covariance
        """
        r = self._whiten(v=v, sigma_inv=sigma_inv, batch=batch, weight=None)
        return numpy.sqrt(numpy.einsum("...i,...i->...", r, r, dtype=numpy.float64))


    def eval_likelihood(self, v: numpy.ndarray, constant: float = 0.0,
                        sigma_inv: float | numpy.ndarray | None = None, batch: int | None = None,
                        out: numpy.ndarray | None = None,
                        weight: numpy.ndarray | None = None) -> float | numpy.ndarray:
        """
        The l2 log likelihood {constant - 0.5 * norm(v)^2}: of a vector {v}, returned as a
        scalar, or of each of the first {batch} rows of {v}, filled into {out} if given and
        returned. {weight} is applied to {v} before {sigma_inv}, so it is only meaningful with
        a diagonal covariance
        """
        r = self._whiten(v=v, sigma_inv=sigma_inv, batch=batch, weight=weight)
        # the sums of squares in double precision, whatever the precision of {v}
        llk = constant - 0.5 * numpy.einsum("...i,...i->...", r, r, dtype=numpy.float64)
        if r.ndim == 1:
            return float(llk)
        if out is None:
            return llk
        out[:r.shape[0]] = llk
        return out


    # implementation details
    def _whiten(self, v: numpy.ndarray, sigma_inv: float | numpy.ndarray | None,
                batch: int | None, weight: numpy.ndarray | None) -> numpy.ndarray:
        """
        The rows of {v}, or {v}, weighed and multiplied by L^T, without modifying {v}
        """
        r = numpy.asarray(v)
        if r.ndim == 2 and batch is not None:
            r = r[:batch]
        if weight is not None:
            r = r * numpy.sqrt(weight)
        # v^T L L^T v = |L^T v|^2, and the rows of v L are the (L^T v)^T
        if sigma_inv is None:
            return r
        if isinstance(sigma_inv, float):
            return r * sigma_inv
        return r @ sigma_inv


# end of file
