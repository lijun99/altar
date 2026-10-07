# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

"""
A thin, {pyre.grid}-backed replacement for the old capsule-based {altar.cuda.matrix}/
{altar.cuda.vector}: this is the "alias" the bayesian state/sampler layer keeps calling
{altar.cuda.matrix(...)}/{altar.cuda.vector(...)} against, so that layer's existing
{.zero()}/{.clone()}/{.copy_from_host(...)}/... call sites keep working unchanged, without
each of them being rewritten to talk to {pyre.grid} directly.

{Array} wraps a {pyre.cuda.managed} grid and adds back the handful of convenience methods
that layer expects. Managed memory is host-visible, so every method here is just a thin
{numpy.asarray(grid)} view over it -- there is no separate host/device copy step the way
there was for the old capsule-based buffers, {copy_from_host}/{copy_to_host} included.
"""

import numpy
import pyre.cuda


# the declaration
class Array:
    """
    A managed-memory buffer, wrapping a {pyre.grid} grid; construct one with the module-level
    {matrix}/{vector} factories below, not directly
    """


    def __init__(self, grid, dtype=None):
        # the grid i wrap
        self._grid = grid
        # its shape and cell type, kept so that asking doesn't wait for the device
        self._shape = tuple(grid.shape)
        self._dtype = _cell(dtype) if dtype is not None else cell(grid)


    # construction
    @classmethod
    def _allocate(cls, shape, dtype):
        """
        Allocate a new, uninitialized grid of the given {shape} and {dtype}
        """
        if isinstance(shape, int):
            shape = (shape,)
        return cls(pyre.cuda.managed(shape=tuple(shape), cell=_cell(dtype)), dtype=dtype)


    @classmethod
    def _wrap(cls, source, dtype=None):
        """
        Allocate a new grid matching {source}'s shape and dtype (or {dtype}), and copy {source} into it
        """
        source = numpy.asarray(source)
        self = cls._allocate(shape=source.shape, dtype=dtype or source.dtype)
        self.copy_from_host(source=source)
        return self


    # shape/dtype accessors
    @property
    def shape(self):
        """
        My shape, as a plain tuple
        """
        return self._shape


    @property
    def rows(self):
        """
        My first extent; only meaningful for a rank-2 (matrix) array
        """
        return self._grid.shape[0]


    @property
    def cols(self):
        """
        My second extent; only meaningful for a rank-2 (matrix) array
        """
        return self._grid.shape[1]


    @property
    def dtype(self):
        """
        My cell type, as the same string spelling ({matrix}/{vector}'s own {dtype}
        parameter accepts it right back, e.g. to build a same-typed buffer elsewhere)
        """
        return self._dtype


    @property
    def grid(self):
        """
        My underlying {pyre.grid} grid, for handing to a cuda extension binding directly
        """
        return self._grid


    # mutators
    def zero(self):
        """
        Fill me with zeroes, in place, on the device
        """
        _grids().zero(self._grid)
        return self


    def fill(self, value):
        """
        Fill me with {value}, in place, on the device
        """
        _grids().zero(self._grid)
        if value != 0:
            self._grid += value
        return self


    def clone(self):
        """
        Make a new array with a duplicate of my cells
        """
        clone = type(self)._allocate(shape=self.shape, dtype=self.dtype)
        _grids().copy(clone._grid, self._grid)
        return clone


    def copy(self, other):
        """
        Overwrite my cells with {other}'s, in place; {other} may be another {Array} or
        anything {numpy.asarray} accepts (e.g. a cpu {altar.matrix}/{altar.vector}, which
        supports the buffer protocol directly); on the device when it is a managed grid like me
        """
        source = self._peer(other)
        if source is not None:
            _grids().copy(self._grid, source)
            return self
        source = other._grid if isinstance(other, Array) else other
        numpy.asarray(self._grid)[...] = numpy.asarray(source)
        return self


    def copy_from_host(self, source):
        """
        Overwrite my cells with {source}'s, in place; {source} is anything {numpy.asarray}
        accepts
        """
        numpy.asarray(self._grid)[...] = numpy.asarray(source)
        return self


    def copy_to_host(self, target=None, type=None):
        """
        Copy my cells out to {target} (anything {numpy.asarray} accepts and can be assigned
        into), or, with {type="numpy"} and no {target}, return a fresh, host-owned
        {numpy.ndarray} instead
        """
        if target is not None:
            numpy.asarray(target)[...] = numpy.asarray(self._grid)
            return target
        # no target: hand back an independent host array (a copy, not the zero-copy managed
        # view itself, so the caller can outlive or mutate it without touching my cells)
        return numpy.asarray(self._grid).copy()


    def mean_sd(self):
        """
        The per-column mean and standard deviation, for a rank-2 (matrix) array
        """
        arr = numpy.asarray(self._grid)
        return arr.mean(axis=0), arr.std(axis=0)


    def sum(self):
        """
        The sum of all my cells, as a plain python scalar (e.g. counting how many cells of an
        int32 flag vector are set), computed on the device
        """
        return _grids().sum(self._grid)


    def cholesky(self, uplo=None):
        """
        Factor me in place as a symmetric positive definite matrix, leaving my own Cholesky
        factor U (self = U^T U) in my row-major upper triangle -- the same convention
        {altar.norms.cuda.L2}/{altar.data.cuda.DataL2} already use and validated against real
        gpu output. {uplo} is accepted, not used: the old capsule api's {uplo=FillModeUpper}
        callers wanted is exactly this method's only (and already validated) behavior.

        Only the upper triangle is written -- the lower triangle is left holding whatever was
        there before (my own original, pre-factored cells, not zeros), matching the
        underlying cusolver call this wraps. A caller that needs the full matrix (e.g. to
        reconstruct {self}, {U.T @ U}, for a sanity check) must mask it first, e.g. with
        {numpy.triu(numpy.asarray(a))}; a caller that only ever reads the upper triangle
        through a triangular blas call (e.g. {cublas.dtrmm}/{dtrmv} with a matching {uplo})
        never needs to.
        """
        import altar.cuda

        cusolver = altar.cuda.cusolver
        cublas = altar.cuda.cublas
        handle = altar.cuda.cusolver_handle()

        n = self.shape[0]
        double = self.dtype == "float64"
        potrf = cusolver.dpotrf if double else cusolver.spotrf
        potrf_buffer_size = cusolver.dpotrf_buffer_size if double else cusolver.spotrf_buffer_size

        dev_info = pyre.cuda.managed(shape=(1,), cell="int32")
        lwork = potrf_buffer_size(handle, cublas.FillMode.LOWER, n, self._grid, n)
        workspace = pyre.cuda.managed(shape=(max(lwork, 1),), cell=self.dtype)
        potrf(handle, cublas.FillMode.LOWER, n, self._grid, n, workspace, lwork, dev_info)
        return self


    # in place arithmetic, for the handful of call sites that scale/accumulate directly
    # (e.g. {sigma_chol *= scaling}, {posterior += beta*data})
    # (on the device, by pyre's grid arithmetic, for a number or a managed grid like me)
    def __imul__(self, scalar):
        self._grid *= scalar
        return self


    def __iadd__(self, other):
        source = self._peer(other)
        if source is not None:
            self._grid += source
            return self
        source = other._grid if isinstance(other, Array) else other
        numpy.asarray(self._grid)[...] += numpy.asarray(source)
        return self


    def __isub__(self, other):
        source = self._peer(other)
        if source is not None:
            self._grid -= source
            return self
        source = other._grid if isinstance(other, Array) else other
        numpy.asarray(self._grid)[...] -= numpy.asarray(source)
        return self


    # implementation details
    def _peer(self, other):
        """
        The grid of {other}, if it is a managed grid with my shape and cell type, so the device
        can combine it with mine; {None} otherwise
        """
        if isinstance(other, Array):
            same = other._shape == self._shape and other._dtype == self._dtype
            return other._grid if same else None
        if getattr(other, "strategy", None) != "managed" or tuple(other.shape) != self._shape:
            return None
        return other if cell(other) == self._dtype else None


    # buffer protocol support
    def __array__(self, dtype=None, copy=None):
        # let {numpy.asarray(array)} work directly, zero-copy; {numpy.array(array)} copies
        arr = numpy.asarray(self._grid)
        if dtype is not None:
            return arr.astype(dtype, copy=bool(copy))
        return arr.copy() if copy else arr


    def __getattr__(self, name):
        # forward anything i don't define myself to the underlying grid (e.g. {strides},
        # {rank}, {writable}, {address}, {__dlpack__})
        return getattr(self._grid, name)


    def __repr__(self):
        return f"<{type(self).__name__} shape={self.shape} dtype={self.dtype}>"


# the cell type {pyre.cuda.managed} wants, from whatever spelling a caller used (a plain
# string like "float64", a numpy dtype, or a numpy scalar type)
def _grids():
    """
    The device copies and fills of {altar.cuda}'s extension, which loads after me
    """
    from .ext import cudaaltar
    return cudaaltar.grids


def _cell(dtype):
    return numpy.dtype(dtype).name


def cell(grid):
    """
    The cell type name of {grid}, an {Array} or a bare managed grid, without waiting for the device
    """
    if isinstance(grid, Array):
        return grid._dtype
    return numpy.dtype(_grids().format(grid)).name


# the module-level factories: the "alias" for the old {altar.cuda.matrix}/{altar.cuda.vector}
def matrix(shape=None, dtype=None, source=None):
    """
    Allocate a (rows x cols) managed-memory matrix, or, with {source} given instead of
    {shape}, one that duplicates {source}'s shape/cells, in {dtype} if given; {dtype}
    defaults to float64 for a fresh allocation, and to {source}'s own otherwise
    """
    if source is not None:
        return Array._wrap(source, dtype=dtype)
    return Array._allocate(shape=shape, dtype=dtype or "float64")


def vector(shape=None, dtype=None, source=None):
    """
    Allocate a managed-memory vector of {shape} cells, or, with {source} given instead of
    {shape}, one that duplicates {source}'s shape/cells, in {dtype} if given; {dtype}
    defaults to float64 for a fresh allocation, and to {source}'s own otherwise
    """
    if source is not None:
        return Array._wrap(source, dtype=dtype)
    return Array._allocate(shape=shape, dtype=dtype or "float64")


# end of file
