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
import typing
from importlib import import_module
# get the package
import altar

# get the protocol
from .Norm import Norm as norm


# the declaration
class Base(altar.component, implements=norm):
    """
    The base class for norms

    Same shape as {altar.distributions.Base}/{altar.models.Base}: I am the one piece pyre
    ever registers, and my only job is to pick my backend implementation once, in {eval} (a
    norm has no separate {initialize}, so the first call is what decides), and forward every
    call to it from then on. My implementation lives in a same-named class in
    {altar.norms.native} (the cpu default) or {altar.norms.cuda}; neither is a pyre component,
    just the numerics.
    """


    # configuration
    @altar.export
    def eval(self, v: typing.Any, sigma_inv: typing.Any = None,
             batch: int | None = None) -> typing.Any:
        """
        Compute the norm of {v}, with or without a covariance matrix: of each row of a
        (samples x observations) batch, as a vector, or, on the cpu, of a single sample, as a
        scalar
        """
        if self._impl is None:
            self._impl = self._makeImpl()
        return self._impl.eval(v=v, sigma_inv=sigma_inv, batch=batch)


    @altar.export
    def eval_likelihood(self, v: typing.Any, constant: float = 0.0, sigma_inv: typing.Any = None,
                        batch: int | None = None, out: typing.Any = None,
                        weight: typing.Any = None) -> typing.Any:
        """
        Compute the log likelihood {constant - 0.5 * norm(v)^2} of each row of a
        (samples x observations) batch, into {out}, allocated if not given, or, on the cpu, of
        a single sample, as a scalar. {weight}, if given, weighs each component of {v}, e.g. a
        0/1 mask of valid observations
        """
        if self._impl is None:
            self._impl = self._makeImpl()
        return self._impl.eval_likelihood(
            v=v, constant=constant, sigma_inv=sigma_inv, batch=batch, out=out, weight=weight)


    def _makeImpl(self) -> typing.Any:
        """
        Build my backend implementation: a same-named class in {native} (the cpu default) or
        {cuda}, picked once, here, based on {altar.backends.active()}
        """
        # my own class name is also my implementation's
        name = type(self).__name__
        # the package it lives in
        backend = "cuda" if altar.backends.active() == "cuda" else "native"
        # reach it and build it
        module = import_module(f"altar.norms.{backend}.{name}")
        return getattr(module, name)()


    # private data
    _impl = None # my backend implementation, chosen once, on first use


# end of file
