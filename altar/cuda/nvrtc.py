# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

"""
Compile CUDA C++ kernels at run time with NVRTC, and launch them on altar's cuda arrays
"""

# externals
import functools
import cuda.core as cc


@functools.cache
def kernels(source, names):
    """
    Compile {source} for the current device, once, and return its kernels {names}, e.g.
    ("predict<float>", "predict<double>"), by name
    """
    device = cc.Device()
    device.set_current()
    options = cc.ProgramOptions(arch=f"sm_{device.arch}", std="c++17")
    code = cc.Program(source, "c++", options).compile("cubin", name_expressions=names)
    return {name: code.get_kernel(name) for name in names}


def launch(kernel, blocks, threads, *args, shared=0):
    """
    Launch {kernel} over {blocks} of {threads} on the default stream, so it is ordered with the
    rest of altar's kernels; {altar.cuda} arrays and grids among {args} are passed as pointers,
    and scalars should carry their C type, e.g. numpy.int32
    """
    arguments = []
    for arg in args:
        grid = getattr(arg, "grid", arg)
        interface = getattr(grid, "__cuda_array_interface__", None)
        arguments.append(arg if interface is None else interface["data"][0])
    config = cc.LaunchConfig(grid=blocks, block=threads, shmem_size=shared)
    cc.launch(cc.LEGACY_DEFAULT_STREAM, config, kernel, *arguments)


def blocks(size, threads):
    """
    The number of blocks of {threads} that cover {size}
    """
    return (size + threads - 1) // threads


# the C++ type of each precision
real = {"float32": "float", "float64": "double"}


# end of file
