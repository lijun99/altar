# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

"""
The cpu (host) implementation of every sampler.

Each module here is named after the sampler it implements, e.g. {Metropolis}, and defines a
plain class of the same name -- not a pyre component, just the algorithm. The registered
component a {.pfg} actually selects lives one level up, in {altar.bayesian.samplers}; it looks
up its implementation here by its own class name, once, at {initialize} time.
"""

# end of file
