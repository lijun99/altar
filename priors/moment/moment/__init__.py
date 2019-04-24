# -*- python -*-
# -*- coding: utf-8 -*-
#
# lijun zhu (ljzhu@gps.caltech.edu)
#
# (c) 2013-2019 parasim inc
# (c) 2010-2019 california institute of technology
# all rights reserved
#


# the package
import altar

# implementations
@altar.foundry(implements=altar.distributions.Distribution, tip="the Moment Magnitude distribution")
def moment():
    # grab the factory
    from .Moment import Moment as moment
    # attach its docstring
    __doc__ = moment.__doc__
    # and return it
    return moment

# end of file
