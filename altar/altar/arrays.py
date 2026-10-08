# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

"""
The array types of the two backends, for annotating the code they share
"""

from __future__ import annotations
import typing
import numpy

if typing.TYPE_CHECKING:
    from altar.cuda.array import Array as DeviceArray

# a host array on the cpu, a managed device array on the gpu
Array: typing.TypeAlias = "numpy.ndarray | DeviceArray"


# end of file
