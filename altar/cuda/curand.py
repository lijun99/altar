# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

"""
altar's own {curand} namespace: everything {pyre.cuda.curand} already has, plus {uniform}/
{gaussian}, the two convenience fills the bayesian sampler layer's random-displacement/
acceptance-draw code needs (curand's own bindings are dtype-specific and take an explicit
cell count; these dispatch on the output's own cell type and fill it completely). A separate
module rather than monkey-patching {pyre.cuda.curand} itself, matching {altar.cuda.cublas}.
"""

import numpy
import pyre.cuda


# the real thing, everything below is layered on top of
_curand = pyre.cuda.curand


def uniform(out, generator=None):
    """
    Fill every cell of {out} with a draw from U(0, 1], dispatching on its cell type; {out}
    may be an {altar.cuda.array.Array} instance or a bare {pyre.grid} grid
    """
    import altar.cuda

    generator = generator if generator is not None else altar.cuda.curand_generator()
    grid = out.grid if hasattr(out, "grid") else out
    n = numpy.asarray(grid).size

    cell = numpy.asarray(grid).dtype.name
    fn = _curand.generate_uniform_double if cell == "float64" else _curand.generate_uniform
    fn(generator, grid, n)
    return out


def gaussian(out, mean=0.0, stddev=1.0, generator=None):
    """
    Fill every cell of {out} with a draw from N(mean, stddev^2), dispatching on its cell
    type; {out} may be an {altar.cuda.array.Array} instance or a bare {pyre.grid} grid.
    curand requires an even cell count for the normal generators; a caller with an odd count
    needs one extra scratch cell of its own (not handled here)
    """
    import altar.cuda

    generator = generator if generator is not None else altar.cuda.curand_generator()
    grid = out.grid if hasattr(out, "grid") else out
    n = numpy.asarray(grid).size

    cell = numpy.asarray(grid).dtype.name
    fn = _curand.generate_normal_double if cell == "float64" else _curand.generate_normal
    fn(generator, grid, n, mean, stddev)
    return out


def __getattr__(name):
    # forward everything else to pyre's own curand
    return getattr(_curand, name)


# end of file
