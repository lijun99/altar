# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-2019 parasim inc
# (c) 2010-2019 california institute of technology
# all rights reserved
#
# Author(s): michael a.g. aïvázis, Lijun Zhu

# pull the extension module; this must exist, so let import errors bubble up
from . import altar as libaltar

# attempt
try:
    # to load the extension with the CUDA support
    from . import cudaaltar as libcudaaltar
# if it fails
except ImportError:
    # no worries
    pass


# end of file
