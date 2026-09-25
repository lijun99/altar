# -*- python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

"""
The cuda implementation of every distribution.

Each module here is named after the distribution it implements, e.g. {Uniform}, and defines a
plain class of the same name -- not a pyre component, just the numerics, mirroring
{altar.distributions.native}. The registered component a {.pfg} actually selects lives one
level up, in {altar.distributions}; it looks up its implementation here by its own class
name, once, at {initialize} time, when {altar.backends.active()} says "cuda".
"""

# end of file
