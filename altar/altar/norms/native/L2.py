# -*- python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
#
# (c) 2013-2025 parasim inc
# (c) 2010-2025 california institute of technology
# all rights reserved
#

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


    def eval_likelihood(self, v, constant=0.0, sigma_inv=None, batch=None, out=None):
        """
        Compute the l2 log likelihood {constant - 0.5 * norm(v)^2}. {out} is cuda only and
        ignored here; cpu always returns the scalar likelihood directly.
        """
        norm = self.eval(v=v, sigma_inv=sigma_inv)
        return constant - 0.5 * norm * norm


    # implementation details
    def _with_covariance(self, v, sigma_inv):
        """
        Compute the L2 norm of the given vector using the given Cholesky decomposed inverse
        covariance matrix
        """
        # we assume {sigma_inv} is Cholesky decomposed, so we can pre-multiply the vector by
        # the lower triangle, and then just take the norm

        # use the lower triangle, no transpose, non-unit diagonal
        if isinstance(sigma_inv, altar.matrix):
            v = altar.blas.dtrmv(
                sigma_inv.lowerTriangular, sigma_inv.opNoTrans, sigma_inv.nonUnitDiagonal,
                sigma_inv, v)
        elif isinstance(sigma_inv, float):
            v *= sigma_inv
        else:
            raise ValueError("L2 norm, sigma_inv should be a matrix or constant")
        # compute the dot product and return it
        return altar.blas.dnrm2(v)


# end of file
