# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# the cublas calls that pyre's bindings don't have, strided batched gemm and dgmm, through
# ctypes, on raw device addresses (offsets into managed grids), with the handle of
# {altar.cuda.cublas_handle}; the library is the libcublas the process already loaded, so
# that the handle is valid for it


# externals
import ctypes
import os

OP_N, OP_T = 0, 1
SIDE_LEFT, SIDE_RIGHT = 0, 1
_lib = None


def library():
    """
    The libcublas of this process
    """
    global _lib
    if _lib is not None:
        return _lib
    import altar.cuda  # noqa: F401, loads pyre's cuda extension and its libcublas
    path = None
    with open("/proc/self/maps") as maps:
        for line in maps:
            name = line.split()[-1] if len(line.split()) >= 6 else ""
            if os.path.basename(name).startswith("libcublas.so"):
                path = name
                break
    _lib = ctypes.CDLL(path or "libcublas.so")
    i, ll, p = ctypes.c_int, ctypes.c_longlong, ctypes.c_void_p
    for prefix in "DS":
        f = getattr(_lib, f"cublas{prefix}gemmStridedBatched")
        f.restype = i
        f.argtypes = [p, i, i, i, i, i, p, p, i, ll, p, i, ll, p, p, i, ll, i]
        g = getattr(_lib, f"cublas{prefix}dgmm")
        g.restype = i
        g.argtypes = [p, i, i, i, p, i, p, i, p, i]
    return _lib


def _check(status, name):
    if status != 0:
        raise RuntimeError(f"{name}: cublas status {status}")


def gemm_strided_batched(handle, precision, transa, transb, m, n, k, alpha, A, lda, strideA,
                         B, ldb, strideB, beta, C, ldc, strideC, count):
    """
    C_i = alpha op(A_i) op(B_i) + beta C_i, i < count, column major; A, B, C are device
    addresses, the strides in cells
    """
    lib = library()
    cell, prefix = (ctypes.c_double, "D") if precision == "float64" else (ctypes.c_float, "S")
    a, b = cell(alpha), cell(beta)
    f = getattr(lib, f"cublas{prefix}gemmStridedBatched")
    _check(f(handle, transa, transb, m, n, k, ctypes.byref(a), A, lda, strideA, B, ldb, strideB,
             ctypes.byref(b), C, ldc, strideC, count), f"cublas{prefix}gemmStridedBatched")
    return


def dgmm(handle, precision, side, m, n, A, lda, x, incx, C, ldc):
    """
    C = diag(x) A (side LEFT) or A diag(x) (RIGHT), column major
    """
    lib = library()
    prefix = "D" if precision == "float64" else "S"
    f = getattr(lib, f"cublas{prefix}dgmm")
    _check(f(handle, side, m, n, A, lda, x, incx, C, ldc), f"cublas{prefix}dgmm")
    return


def address(array, offset=0, precision="float64"):
    """
    The device address of cell {offset} of an {altar.cuda} array or grid
    """
    grid = array.grid if hasattr(array, "grid") else array
    return grid.address + offset * (8 if precision == "float64" else 4)


# end of file
