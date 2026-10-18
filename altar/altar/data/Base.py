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
from importlib import import_module
# get the package
import altar

# get the protocol
from .DataObs import DataObs as data


# the declaration
class Base(altar.component, implements=data):
    """
    The base class for observed data

    Same shape as {altar.norms.Base}: I am the one piece pyre ever registers, and my only
    job is to pick my backend implementation once, in {initialize}, and forward every call to
    it from then on. My implementation lives in a same-named class in {altar.data.native} (the
    cpu default) or {altar.data.cuda}; neither is a pyre component, just the numerics.
    """


    # configuration
    @altar.export
    def initialize(self, application):
        """
        Initialize data from the model
        """
        # pick my backend implementation, once
        self._impl = self._makeImpl()
        # and let it initialize itself
        self._impl.initialize(application=application)
        # all done
        return self


    def _makeImpl(self):
        """
        Build my backend implementation: a same-named class in {native} (the cpu default) or
        {cuda}, picked once, here, based on {altar.backends.active()}
        """
        # my own class name is also my implementation's
        name = type(self).__name__
        # the package it lives in
        backend = "cuda" if altar.backends.active() == "cuda" else "native"
        # reach it and build it
        module = import_module(f"altar.data.{backend}.{name}")
        return getattr(module, name)()


    @altar.export
    def eval_likelihood(self, prediction, likelihood, residual=True, batch=None, whitened=True):
        """
        Compute the data log likelihood for {prediction} and deposit it in {likelihood};
        {whitened=False} marks {prediction} as a raw model prediction, without the data
        covariance merged into it
        """
        return self._impl.eval_likelihood(
            prediction=prediction, likelihood=likelihood, residual=residual, batch=batch,
            whitened=whitened)


    def update_covariance(self, cp=None):
        """
        Update the data covariance with {cp} (a model-uncertainty contribution), and refresh
        everything that's derived from it (the inverse, the normalization, the merged data)
        """
        return self._impl.update_covariance(cp=cp)


    def release_cd(self):
        """
        Release {cd_inv}; cuda only, a no-op on the cpu backend that doesn't define it
        """
        release = getattr(self._impl, "release_cd", None)
        return release() if release is not None else None


    def observed(self):
        """
        The raw observed data, before any covariance is merged into it, as a numpy vector
        """
        return self._impl.observed()


    def sigma(self):
        """
        The standard deviation of each observation, sqrt(diag(Cd)), as a numpy vector
        """
        return self._impl.sigma()


    def sigma_chi(self):
        """
        The standard deviation of each observation under C_chi = C_d + C_p, sqrt(diag(C_chi)),
        as a numpy vector; the same as {sigma} without a C_p
        """
        return self._impl.sigma_chi()


    def covariance(self):
        """
        The covariance in effect, C_d or C_chi = C_d + C_p: a numpy (observations x observations)
        array, or a float, the common variance, when it is a constant times the identity
        """
        return self._impl.covariance()


    @property
    def dataobs(self):
        """
        The observed data vector
        """
        return self._impl.dataobs


    @property
    def dataobs_batch(self):
        """
        A batch of duplicated observations, one copy per sample; cuda only
        """
        return self._impl.dataobs_batch


    @property
    def mask(self):
        """
        The boolean mask of valid observations, as a numpy vector; {None} if all are valid
        """
        return self._impl.mask


    @property
    def cd_inv(self):
        """
        The inverse of the data covariance, in Cholesky decomposed form
        """
        return self._impl.cd_inv


    @property
    def normalization(self):
        """
        The l2 likelihood normalization constant
        """
        return self._impl.normalization


    # private data
    _impl = None # my backend implementation, chosen once, in {initialize}


# end of file
