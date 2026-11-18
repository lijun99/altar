# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# externals
from __future__ import annotations
import numpy
import scipy.linalg


class Kronecker:
    """
    A data covariance separable in space and time, per component: the observations ordered as
    (component, station, bin), the bin fastest, with Cov(x[c, s, i], x[c, s', j]) =
    S[c, s, s'] R[c, i, j] and no correlation across components, i.e. Cd = blockdiag_c(S_c ⊗ R_c)

    With S_c = L_S L_S^T and R_c = L_R L_R^T, W = blockdiag_c(L_S^-1 ⊗ L_R^-1) whitens, W^T W =
    Cd^-1, as L_S^-1 X L_R^-T on each component X of a data vector; nothing of the size of Cd is
    ever formed
    """


    def __init__(self, S: numpy.ndarray, R: numpy.ndarray) -> None:
        S = numpy.asarray(S, dtype=float)
        R = numpy.asarray(R, dtype=float)
        if S.ndim != 3 or R.ndim != 3 or S.shape[0] != R.shape[0] or S.shape[1] != S.shape[2] \
                or R.shape[1] != R.shape[2]:
            raise ValueError(f"S (components, stations, stations) and R (components, bins, bins); "
                             f"got {S.shape} and {R.shape}")
        self.S, self.R = S, R
        self.components, self.stations, self.bins = S.shape[0], S.shape[1], R.shape[1]
        self.observations = self.components * self.stations * self.bins
        # the lower cholesky factors, which fail unless positive definite
        self.LS = numpy.array([numpy.linalg.cholesky(s) for s in S])
        self.LR = numpy.array([numpy.linalg.cholesky(r) for r in R])
        # log |Cd| = sum_c bins log |S_c| + stations log |R_c|
        self.logdet = float(sum(2 * self.bins * numpy.log(numpy.diag(ls)).sum()
                                + 2 * self.stations * numpy.log(numpy.diag(lr)).sum()
                                for ls, lr in zip(self.LS, self.LR)))
        return


    def whiten(self, rows: numpy.ndarray, transpose: bool = False, chunk: int = 512) -> numpy.ndarray:
        """
        W x for each row x of {rows}, or W^T x with {transpose}, in a new array of their dtype;
        {chunk} rows at a time
        """
        rows = numpy.asarray(rows)
        single = rows.ndim == 1
        X = numpy.atleast_2d(rows)
        if X.shape[-1] != self.observations:
            raise ValueError(f"rows of {self.observations} observations; got {X.shape[-1]}")
        out = numpy.empty(X.shape, dtype=X.dtype if X.dtype.kind == "f" else float)
        for start in range(0, X.shape[0], chunk):
            block = numpy.asarray(X[start:start + chunk], dtype=float)
            out[start:start + chunk] = self._whiten(block, transpose)
        return out[0] if single else out


    def variance(self) -> numpy.ndarray:
        """
        The variance of each observation, diag(Cd)
        """
        return numpy.concatenate([numpy.outer(numpy.diag(s), numpy.diag(r)).ravel()
                                  for s, r in zip(self.S, self.R)])


    def dense(self) -> numpy.ndarray:
        """
        Cd as a dense (observations x observations) matrix
        """
        return scipy.linalg.block_diag(*[numpy.kron(s, r) for s, r in zip(self.S, self.R)])


    def draw(self, rng: numpy.random.Generator, count: int | None = None) -> numpy.ndarray:
        """
        Noise of covariance Cd, {count} rows of it, or one vector
        """
        n = 1 if count is None else count
        Z = rng.standard_normal((n, self.components, self.stations, self.bins))
        X = numpy.einsum("cst,nctb,cib->ncsi", self.LS, Z, self.LR).reshape(n, -1)
        return X[0] if count is None else X


    @property
    def factor(self) -> Factor:
        """
        L = W^T, with Cd^-1 = L L^T, as {altar.data.DataL2.cd_inv}: x @ L whitens a row x
        """
        return Factor(self)


    # implementation details
    def _whiten(self, X: numpy.ndarray, transpose: bool) -> numpy.ndarray:
        """
        L_S^-1 X_c L_R^-T, or L_S^-T X_c L_R^-1, for each component X_c of each row of {X}
        """
        n = X.shape[0]
        C, ns, nt = self.components, self.stations, self.bins
        X = X.reshape(n, C, ns, nt)
        Y = numpy.empty_like(X)
        solve = scipy.linalg.solve_triangular
        for c in range(C):
            # along the stations: (ns, n*nt)
            A = X[:, c].transpose(1, 0, 2).reshape(ns, n * nt)
            A = solve(self.LS[c], A, lower=True, trans="T" if transpose else "N", check_finite=False)
            # then along the bins: (nt, n*ns)
            B = A.reshape(ns, n, nt).transpose(2, 1, 0).reshape(nt, n * ns)
            B = solve(self.LR[c], B, lower=True, trans="T" if transpose else "N", check_finite=False)
            Y[:, c] = B.reshape(nt, n, ns).transpose(1, 2, 0)
        return Y.reshape(n, -1)


class Factor:
    """
    The factor L = W^T of the inverse of a {Kronecker} covariance, Cd^-1 = L L^T, as a matrix on
    the right of rows: x @ L whitens each row x, x @ L.T applies W^T, so that x @ L @ L.T = x Cd^-1
    """

    # numpy defers {rows @ factor} to my {__rmatmul__}
    __array_ufunc__ = None


    def __init__(self, covariance: Kronecker, transposed: bool = False) -> None:
        self.covariance = covariance
        self.transposed = transposed
        self.shape = (covariance.observations, covariance.observations)
        return


    def whiten(self, rows: numpy.ndarray) -> numpy.ndarray:
        """
        rows @ L: W x for each row x
        """
        return self.covariance.whiten(rows)


    @property
    def T(self) -> Factor:
        return Factor(self.covariance, transposed=not self.transposed)


    def __rmatmul__(self, rows: numpy.ndarray) -> numpy.ndarray:
        # x @ L = (W x^T)^T, x @ L^T = (W^T x^T)^T
        return self.covariance.whiten(rows, transpose=self.transposed)


    def __array__(self, dtype=None, copy=None) -> numpy.ndarray:
        # dense, for small problems and checks: L = W^T
        W = self.covariance.whiten(numpy.eye(self.shape[0])).T
        L = W if self.transposed else W.T
        return L if dtype is None else L.astype(dtype)


# end of file
