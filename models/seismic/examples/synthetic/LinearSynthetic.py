#!/usr/bin/env python3
# -*- python -*-
# -*- coding: utf-8 -*-
#
#
# (c) 2013-2021 parasim inc
# (c) 2010-2021 california institute of technology
# all rights reserved
#
# Author(s): Lijun Zhu

# the framework
import altar
# externals
import h5py
import numpy
import os

# app
class LinearSynthetic(altar.application, family="altar.application.linearsynthetic"):
    """
    An application to generate a synthetic linear model
    """

    # user configurable state
    case = altar.properties.path(default='./')
    case.doc = "the directory of input files, default is current directory"

    parameters  = altar.properties.int(default=16)
    parameters.doc = "number of parameters"

    random = altar.properties.bool(default=False)
    random.doc = "whether to generate random parameters"


    # protocol obligation
    @altar.export
    def main(self, *args, **kwds):
        """
        The main entry point
        """

        # green function as identity matrix
        green = numpy.identity(n=self.parameters)
        # generate theta
        if self.random :
            theta = numpy.random.rand(self.parameters,1)
        else:
            theta = numpy.ndarray(shape=(self.parameters,1))
            theta.fill(1)

        # d = G \theta
        data = numpy.matmul(green, theta)

        # save them to text
        numpy.savetxt('synthetic/green.txt', green)
        numpy.savetxt('synthetic/theta.txt', theta)
        numpy.savetxt('synthetic/data.txt', data)

        # all done
        return 0

    # pyre framework hooks
    # support for the help system
    def pyre_banner(self):
        """
        Place the application banner in the {info} channel
        """
        # show the package header
        return altar.meta.header

    # interactive session management
    def pyre_interactiveSessionContext(self, context):
        """
        Go interactive
        """
        # protect against bad context
        if context is None:
            # by initializing an empty one
            context = {}

        # add some symbols
        context["linearsynthetic"] = linearsynthetic  # my package

        # and chain up
        return super().pyre_interactiveSessionContext(context=context)


# bootstrap
if __name__ == "__main__":
    # instantiate
    app = LinearSynthetic(name="linearsynthetic")
    # invoke
    status = app.run()
    # share
    raise SystemExit(status)


# end of file
