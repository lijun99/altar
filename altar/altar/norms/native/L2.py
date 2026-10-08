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
        Compute the l2 log likelihood {constant - 0.5 * norm(v)^2}: of a vector {v}, returned
        as a scalar, or of each of the first {batch} rows of a (samples x observations) {v},
        filled into {out} if given and returned. {weight} is applied to {v} before
        {sigma_inv}, so it is only meaningful with a diagonal covariance
        """
        if numpy.ndim(v) == 2:
            return self._eval_likelihood_batched(
                v=v, constant=constant, sigma_inv=sigma_inv, batch=batch, out=out, weight=weight)
        if weight is not None:
            v = v.clone()
            numpy.asarray(v)[:] *= numpy.sqrt(numpy.asarray(weight))
        norm = self.eval(v=v, sigma_inv=sigma_inv)
        return constant - 0.5 * norm * norm


    # implementation details
    def _eval_likelihood_batched(self, v, constant, sigma_inv, batch, out, weight):
        """
        The log likelihoods of the first {batch} rows of {v}, without modifying {v}
        """
        r = numpy.asarray(v)
        batch = r.shape[0] if batch is None else batch
        r = r[:batch]
        if weight is not None:
            r = r * numpy.sqrt(numpy.asarray(weight))
        # each row v^T L L^T v = |L^T v|^2, with the rows of v L
        if isinstance(sigma_inv, float):
            r = r * sigma_inv
        elif sigma_inv is not None:
            r = r @ numpy.tril(numpy.asarray(sigma_inv))
        llk = constant - 0.5 * numpy.einsum("ij,ij->i", r, r)
        if out is None:
            return llk
        numpy.asarray(out)[:batch] = llk
        return out

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
