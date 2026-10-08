# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

"""
Launch cuTile ({cuda.tile}) kernels on altar's cuda arrays
"""

# externals
import cuda.tile as ct


class _Interface:
    """
    Expose a managed grid to cuTile only through the cuda array interface: its dlpack device
    is kDLCUDAManaged, which cuTile rejects
    """
    def __init__(self, grid):
        self.grid = grid
        self.__cuda_array_interface__ = grid.__cuda_array_interface__


def launch(kernel, blocks, *args):
    """
    Launch {kernel} over {blocks} on the default stream, so it is ordered with the rest of
    altar's kernels; {altar.cuda} arrays and grids among {args} are passed as device arrays
    """
    arguments = []
    for arg in args:
        grid = getattr(arg, "grid", arg)
        arguments.append(_Interface(grid) if hasattr(grid, "__cuda_array_interface__") else arg)
    # pad the launch grid to three dimensions
    blocks = tuple(blocks) + (1,) * (3 - len(blocks))
    ct.launch(0, blocks, kernel, tuple(arguments))


def constant(constants, index):
    """
    Read entry {index} of the device vector {constants} as a scalar, in tile code: python
    floats, as kernel arguments or literals, are float32 in cuTile, so double precision
    constants have to come from the device
    """
    return ct.reshape(ct.load(constants, index=(index,), shape=(1,)), ())


def blocks(size, tile):
    """
    The number of {tile} sized blocks that cover {size}
    """
    return (size + tile - 1) // tile


# end of file
