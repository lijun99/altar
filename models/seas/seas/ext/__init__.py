# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-2021 parasim inc
# (c) 2010-2021 california institute of technology
# all rights reserved
#


# attempt
try:
    # to load the extension with the CUDA support
    from . import cudaseas as libcuseas
# if it fails
except ImportError:
    # no worries
    pass


# end of file
