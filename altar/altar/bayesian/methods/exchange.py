# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

"""
Moving the chains of a step among the processes of an mpi run
"""

from __future__ import annotations
import typing
import numpy


def collect(array: numpy.ndarray, communicator: typing.Any,
            destination: int) -> numpy.ndarray | None:
    """
    The {array} of every process, stacked in rank order at {destination}; None elsewhere
    """
    parts = communicator.gatherObject(item=numpy.asarray(array), root=destination)
    return None if parts is None else numpy.concatenate(parts)


def excerpt(target: numpy.ndarray, array: numpy.ndarray | None, source: int,
            communicator: typing.Any) -> numpy.ndarray:
    """
    Fill {target} with my share of the {array} held by {source}, split by rank into equal parts
    """
    parts = None
    if communicator.rank == source:
        parts = numpy.array_split(array, communicator.size)
    target[...] = communicator.scatterObject(items=parts, root=source)
    return target


# end of file
