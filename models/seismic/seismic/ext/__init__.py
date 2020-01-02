# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-2020 parasim inc
# (c) 2010-2020 california institute of technology
# all rights reserved
#
# Author(s): Lijun Zhu


# attempt
try:
    # to load the extension with the CUDA support
    from . import cudaseismic as libcuseismic
# if it fails
except ImportError:
    # no worries
    pass


# end of file
