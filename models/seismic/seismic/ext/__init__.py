# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#


# attempt
try:
    # to load the extension with the CUDA support
    from . import cudaseismic as libcuseismic
# if it fails
except ImportError:
    # no worries
    pass


# end of file
