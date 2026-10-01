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
import numpy
# get the package
import altar


# the declaration
class L2:
    """
    The cpu implementation of the L2 norm
    """


    def eval(self, v, sigma_inv=None, batch=None):
        """
        Compute the L2 norm of the given vector, with or without a covariance matrix
        """
        # if we have a covariance matrix
        if sigma_inv is not None:
            # use the specialized implementation
            return self._with_covariance(v=v, sigma_inv=sigma_inv)
        # otherwise, compute the norm and return it
        return altar.blas.dnrm2(v)


    def eval_likelihood(self, v, constant=0.0, sigma_inv=None, batch=None, out=None,
                        weight=None):
        """
        Compute the l2 log likelihood {constant - 0.5 * norm(v)^2}. {out} is cuda only and
        ignored here; cpu always returns the scalar likelihood directly. {weight} is applied
        to {v} before {sigma_inv}, so it is only meaningful with a diagonal covariance
        """
        if weight is not None:
            v = v.clone()
            numpy.asarray(v)[:] *= numpy.sqrt(numpy.asarray(weight))
        norm = self.eval(v=v, sigma_inv=sigma_inv)
        return constant - 0.5 * norm * norm


    # implementation details
    def _with_covariance(self, v, sigma_inv):
        """
        Compute the L2 norm of the given vector using the given Cholesky decomposed inverse
        covariance matrix
        """
        # {sigma_inv} holds L, the lower Cholesky factor of the inverse covariance, L L^T, so
        # v^T L L^T v = |L^T v|^2: pre-multiply by L^T, then just take the norm
        if isinstance(sigma_inv, altar.matrix):
            v = altar.blas.dtrmv(
                sigma_inv.lowerTriangular, sigma_inv.opTrans, sigma_inv.nonUnitDiagonal,
                sigma_inv, v)
        elif isinstance(sigma_inv, float):
            v *= sigma_inv
        else:
            raise ValueError("L2 norm, sigma_inv should be a matrix or constant")
        # compute the dot product and return it
        return altar.blas.dnrm2(v)


# end of file
