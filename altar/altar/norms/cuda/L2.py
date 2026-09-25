# -*- python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# get the package
import altar.cuda


# the declaration
class L2:
    """
    The cuda implementation of the L2 norm

    Unlike the cpu side, {v} here is a full (samples x observations) grid, not one sample at
    a time, and {eval}/{eval_likelihood} fill a (samples,) grid of per-sample results.
    """


    def eval(self, v, sigma_inv=None, batch=None, out=None):
        """
        Fill {out} (allocating one if not given) with the l2 norm of each of the first
        {batch} rows of {v}, with or without a covariance matrix
        """
        # {v}/{out} may be a bare {pyre.grid} grid or an {altar.cuda.array.Array} (the state
        # layer's {altar.cuda.matrix}/{altar.cuda.vector} instances); the extension boundary
        # below needs the bare grid
        v = self._grid(v)
        out = self._grid(out) if out is not None else None

        samples = v.shape[0]
        batch = batch or samples
        out = out if out is not None else pyre_grid_managed(shape=(samples,), cell=self._cell(v))

        # if a covariance matrix is given, apply it to {v} first, in place
        if sigma_inv is not None:
            self._apply_covariance(v=v, sigma_inv=self._grid(sigma_inv))

        # the extension dispatches on {v}'s own cell type internally, so the same call
        # handles float32 and float64 alike
        altar.cuda.libcudaaltar.norms.cudaL2_norm(v, out, batch)
        return out


    def eval_likelihood(self, v, constant=0.0, sigma_inv=None, batch=None, out=None):
        """
        Fill {out} with the l2 log likelihood {constant - 0.5 * norm(v)^2} of each of the
        first {batch} rows of {v}
        """
        v = self._grid(v)
        out = self._grid(out) if out is not None else None

        samples = v.shape[0]
        batch = batch or samples
        out = out if out is not None else pyre_grid_managed(shape=(samples,), cell=self._cell(v))

        if sigma_inv is not None:
            self._apply_covariance(v=v, sigma_inv=self._grid(sigma_inv))

        altar.cuda.libcudaaltar.norms.cudaL2_normllk(v, out, batch, constant)
        return out


    @staticmethod
    def _grid(buffer):
        """
        The bare {pyre.grid} grid underneath {buffer}, for handing to a cuda extension
        binding: {buffer} may already be one, or it may be an {altar.cuda.array.Array} (the
        state layer's {altar.cuda.matrix}/{altar.cuda.vector} instances)
        """
        return buffer.grid if hasattr(buffer, "grid") else buffer


    # implementation details
    def _apply_covariance(self, v, sigma_inv):
        """
        Apply the Cholesky-decomposed inverse covariance matrix {sigma_inv} (observations x
        observations, upper triangle) to {v} (samples x observations), in place:
        v <- v @ sigma_inv, row-major -- {v}'s own rows, whitened by the covariance factor

        trmm's "A" operand is always the triangular one (here {sigma_inv}) and "B" the
        general one (here {v}); a row-major "C = B @ A" (side right) becomes, read as
        column-major, "C^T = A^T @ B^T" -- side left, with upper/lower swapped since a
        row-major buffer read as column-major is its own transpose. Unlike gemm's swap, trmm
        never swaps which operand is passed as cublas's A vs B, only the side/uplo flags and
        the extents.
        """
        cublas = altar.cuda.cublas
        handle = altar.cuda.cublas_handle()
        trmm = cublas.dtrmm if self._cell(v) == "float64" else cublas.strmm

        observations = v.shape[1]
        samples = v.shape[0]

        trmm(
            handle,
            cublas.SideMode.LEFT, cublas.FillMode.LOWER, cublas.Operation.N,
            cublas.DiagType.NON_UNIT,
            observations, samples, 1.0,
            sigma_inv, observations,
            v, observations,
            v, observations,
        )
        return v


    @staticmethod
    def _cell(grid):
        """
        The cell type name (e.g. "float64") a pyre.grid.Grid was built with; grids support
        the buffer protocol, so a memoryview is the cheapest way to ask from python
        """
        format = memoryview(grid).format
        return "float64" if format == "d" else "float32"


def pyre_grid_managed(shape, cell):
    """
    Allocate a fresh grid of cuda managed memory; a thin indirection so this module doesn't
    need a hard import of {pyre.grid} at module-load time before cuda is known to be active
    """
    import pyre.grid
    return pyre.grid.managed(shape=shape, cell=cell)


# end of file
