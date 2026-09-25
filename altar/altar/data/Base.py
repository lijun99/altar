# -*- python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
#
# (c) 2013-2025 parasim inc
# (c) 2010-2025 california institute of technology
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
    def eval_likelihood(self, prediction, likelihood, residual=True, batch=None):
        """
        Compute the data log likelihood for {prediction} and deposit it in {likelihood}
        """
        return self._impl.eval_likelihood(
            prediction=prediction, likelihood=likelihood, residual=residual, batch=batch)


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
