# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#


# externals
import numpy
# the package
import altar
import altar.cuda
from altar.cuda import nvrtc
from . import cublasx


# the kernels, in CUDA C++, compiled by NVRTC the first time they are needed; the vectors of
# the conjugate gradients are (rows x k x J), row major: k right-hand sides interleaved, so that
# the products with the Green's functions are single gemms
SOURCE = r"""
// out[s] (+)= a_s . b_s, a_s the cells [n, s, :] of a (rows x k x J) array; one block per s,
// the block size a power of two
template <typename real>
__global__ void dot_rows(const real * a, const real * b, int rows, int k, int J, real * out,
                         int accumulate)
{
    extern __shared__ double sums[];
    const unsigned long long n = (unsigned long long) rows * J;
    const unsigned long long stride = (unsigned long long) k * J;
    double s = 0;
    for (unsigned long long i = threadIdx.x; i < n; i += blockDim.x) {
        unsigned long long cell = (i / J) * stride + blockIdx.x * (unsigned long long) J + i % J;
        s += a[cell] * b[cell];
    }
    sums[threadIdx.x] = s;
    __syncthreads();
    for (int half = blockDim.x / 2; half > 0; half /= 2) {
        if (threadIdx.x < half) sums[threadIdx.x] += sums[threadIdx.x + half];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = (accumulate ? out[blockIdx.x] : 0) + sums[0];
}

// y += s c[s] x, the coefficient of each right-hand side s
template <typename real>
__global__ void axpy_rows(real * y, const real * x, const real * c, real s, int J, int k,
                          unsigned long long total)
{
    for (unsigned long long i = blockIdx.x * (unsigned long long) blockDim.x + threadIdx.x;
         i < total; i += (unsigned long long) gridDim.x * blockDim.x)
        y[i] += s * c[(i / J) % k] * x[i];
}

// p = z + c[s] p
template <typename real>
__global__ void xpby_rows(real * p, const real * z, const real * c, int J, int k,
                          unsigned long long total)
{
    for (unsigned long long i = blockIdx.x * (unsigned long long) blockDim.x + threadIdx.x;
         i < total; i += (unsigned long long) gridDim.x * blockDim.x)
        p[i] = z[i] + c[(i / J) % k] * p[i];
}

// out[s] = num[s] / den[s]
template <typename real>
__global__ void ratio(const real * num, const real * den, real * out, int k)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < k) out[i] = num[i] / den[i];
}

// y[n, s, j] += x[n, s, j] d[j]
template <typename real>
__global__ void add_scaled_columns(real * y, const real * x, const real * d, int J,
                                   unsigned long long total)
{
    for (unsigned long long i = blockIdx.x * (unsigned long long) blockDim.x + threadIdx.x;
         i < total; i += (unsigned long long) gridDim.x * blockDim.x)
        y[i] += x[i] * d[i % J];
}

// y[n, s, j] /= d[n, j]
template <typename real>
__global__ void divide(real * y, const real * d, int J, int k, unsigned long long total)
{
    const unsigned long long stride = (unsigned long long) k * J;
    for (unsigned long long i = blockIdx.x * (unsigned long long) blockDim.x + threadIdx.x;
         i < total; i += (unsigned long long) gridDim.x * blockDim.x)
        y[i] /= d[(i / stride) * J + i % J];
}

// y[n, s, j] += d[n] x[n, s, j]
template <typename real>
__global__ void add_scaled_rows(real * y, const real * x, const real * d, int J, int k,
                                unsigned long long total)
{
    const unsigned long long stride = (unsigned long long) k * J;
    for (unsigned long long i = blockIdx.x * (unsigned long long) blockDim.x + threadIdx.x;
         i < total; i += (unsigned long long) gridDim.x * blockDim.x)
        y[i] += d[i / stride] * x[i];
}
"""

KERNELS = ("dot_rows", "axpy_rows", "xpby_rows", "ratio", "add_scaled_columns", "divide",
           "add_scaled_rows")
THREADS = 256


# declaration
class Operator:
    """
    The posterior precision of {altar.models.gnss.tdslip.Slip} at beta, as its {Operator}, with
    the solves and the perturbations on the device: cublas gemm and a few NVRTC kernels, double
    precision, all the right-hand sides at once

    The conjugate gradients run in the coordinates X = U Y, Xi_c = E_c Yi_c of the
    preconditioner (U^T P U = I; E_c^T S_c^-1 E_c = diag(lamS_c), E_c orthogonal), where the
    operator is
        ( (G U)^T W + Y diag(dinv),  [E_c^T W_c + diag(lamS_c) Yi_c Pm_c]_c ),
        T = G U Y + E Yi,  W_b = beta M_b T_b,
    and the preconditioner (Y V / denom) V^T: no product with the dense 2N x 2N prior per
    iteration. In exact arithmetic, the iterates are those of the host's {pcg}, and so is its
    stopping rule, sqrt(r^T z / r0^T z0), which doesn't depend on the coordinates. The vectors
    are (rows x k x J), the k right-hand sides interleaved, so that the products with G U and
    E_c are single gemms over all of them; the noise groups are contiguous ranges of the blocks
    once they are permuted by {perm}.

    The perturbations, N(0, A), are drawn on the host's generator and formed on the device: of
    the data, G^T e and e, e_b = (beta M_b)^1/2 z_b; of the prior, (P U) z diag(dinv)^1/2 for X
    (P U (P U)^T = P) and E_c diag(lamS_c)^1/2 z Rm_c^T for Xi_c
    """

    bounded = False


    def __init__(self, host):
        s = host.slip
        self.host = host
        self.size = host.size
        self.N2, self.J, self.B = 2 * host.N, host.J, host.B
        self.xi = bool(s.groups)
        self.perm = (numpy.concatenate([g["idx"] for g in s.groups]) if self.xi
                     else numpy.arange(self.B))
        if not numpy.array_equal(numpy.sort(self.perm), numpy.arange(self.B)):
            raise ValueError("the noise groups don't partition the blocks")
        Gp = s.G[self.perm]
        upload = lambda a: altar.cuda.vector(source=numpy.ascontiguousarray(a, dtype=float).ravel())
        self.G = upload(Gp)
        self.GU = upload(Gp @ host.U)
        self.PU = upload(s.P @ host.U)
        self.U = upload(host.U)
        self.V = upload(host.V)
        self.denom = upload(host.denom)
        self.dinv = upload(s.dinv)
        self.sqrt_dinv = numpy.sqrt(s.dinv)
        self.M = upload(host.bM[self.perm])
        self.Msqrt = upload(host.Msqrt[self.perm])
        self.groups = []
        start = 0
        for g, p in zip(s.groups, host.pre):
            nb = g["idx"].size
            self.groups.append({"start": start, "nb": nb, "E": upload(g["E"]),
                                "lam": upload(g["lamS"]),
                                "sqrt_lam": numpy.sqrt(numpy.maximum(g["lamS"], 0)),
                                "Pm": upload(g["Pm"]), "Rm": upload(g["Rm"]),
                                "V": upload(p["V"]), "denom": upload(p["denom"])})
            start += nb
        self.kernels = nvrtc.kernels(SOURCE, tuple(f"{k}<double>" for k in KERNELS))
        cublasx.library()
        self.buffers = {}
        return


    # the protocol of the operators, see {altar.bayesian.samplers.Perturbation}
    def apply(self, x):
        return self.host.apply(x)


    def rhs(self):
        return self.host.rhs()


    def precondition(self, r):
        return self.host.precondition(r)


    def solve(self, rhs, tolerance, max_iterations):
        """
        A^-1 b for each row b of {rhs} by preconditioned conjugate gradients on the device; the
        solutions, the iterations and the largest final residual sqrt(r^T z / r0^T z0)
        """
        rhs = numpy.atleast_2d(rhs)
        k = rhs.shape[0]
        J, N2, B = self.J, self.N2, self.B
        nx, ni, kJ = N2 * J, B * J, k * J
        buf = self._buffers(k)
        launch = self._launch
        # r = (U^T R, E^T Ri), from the rows of {rhs} to the interleaved coordinates of the
        # preconditioner
        self._upload(buf["t1"], rhs[:, :nx])
        self._gemm(True, False, N2, J, N2, 1.0, self.U, N2, 0, buf["t1"], J, nx, 0.0,
                   buf["r"], kJ, J, k)
        buf["x"].zero()
        if self.xi:
            self._upload(buf["W"], rhs[:, nx:].reshape(k, B, J)[:, self.perm])
            for g in self.groups:
                nb, o = g["nb"], g["start"]
                self._gemm(True, False, nb, J, nb, 1.0, g["E"], nb, 0, (buf["W"], o * J), J, ni,
                           0.0, (buf["ri"], o * kJ), kJ, J, k)
            buf["xi"].zero()
        self._precondition(buf, k)
        buf["p"].copy(buf["z"])
        if self.xi:
            buf["pi"].copy(buf["zi"])
        self._dot(buf, "r", "z", "ri", "zi", "rz", k)
        rz0 = self._download(buf["rz"], k)
        scale = numpy.where(rz0 > 0, rz0, 1.0)
        residual = float(numpy.sqrt(numpy.abs(rz0 / scale)).max())
        iterations = 0
        parts = [("x", "r", "p", "ap", nx)] + ([("xi", "ri", "pi", "api", ni)] if self.xi else [])
        while residual >= tolerance and iterations < max_iterations:
            self._apply(buf, k)
            self._dot(buf, "p", "ap", "pi", "api", "pap", k)
            launch("ratio", k, buf["rz"], buf["pap"], buf["alpha"], numpy.int32(k))
            for x, r, p, ap, n in parts:
                total = numpy.uint64(n * k)
                launch("axpy_rows", n * k, buf[x], buf[p], buf["alpha"], numpy.float64(1),
                       numpy.int32(J), numpy.int32(k), total)
                launch("axpy_rows", n * k, buf[r], buf[ap], buf["alpha"], numpy.float64(-1),
                       numpy.int32(J), numpy.int32(k), total)
            self._precondition(buf, k)
            self._dot(buf, "r", "z", "ri", "zi", "rznew", k)
            rz = self._download(buf["rznew"], k)
            residual = float(numpy.sqrt(numpy.abs(rz / scale)).max())
            iterations += 1
            if residual < tolerance:
                break
            launch("ratio", k, buf["rznew"], buf["rz"], buf["beta"], numpy.int32(k))
            for _, _, p, _, n in parts:
                launch("xpby_rows", n * k, buf[p], buf["z" if p == "p" else "zi"], buf["beta"],
                       numpy.int32(J), numpy.int32(k), numpy.uint64(n * k))
            buf["rz"].copy(buf["rznew"])
        # back to the rows: X = U Y, Xi_c = E_c Yi_c
        self._gemm(False, False, N2, J, N2, 1.0, self.U, N2, 0, buf["x"], kJ, J, 0.0,
                   buf["t1"], J, nx, k)
        out = numpy.empty((k, self.size))
        out[:, :nx] = self._download(buf["t1"], nx * k).reshape(k, nx)
        if self.xi:
            for g in self.groups:
                nb, o = g["nb"], g["start"]
                self._gemm(False, False, nb, J, nb, 1.0, g["E"], nb, 0, (buf["xi"], o * kJ), kJ,
                           J, 0.0, (buf["W"], o * J), J, ni, k)
            Xi = numpy.empty((k, B, J))
            Xi[:, self.perm] = self._download(buf["W"], ni * k).reshape(k, B, J)
            out[:, nx:] = Xi.reshape(k, ni)
        return out, iterations, residual


    def perturb(self, rows, rng):
        """
        {rows} draws of N(0, A), the normals from {rng}, formed on the device, as rows
        """
        k = rows
        J, N2, B = self.J, self.N2, self.B
        nx, ni = N2 * J, B * J
        buf = self._buffers(k)
        # the data: e_b = (beta M_b)^1/2 z_b, in the permuted order (the z are exchangeable)
        self._upload(buf["T"], rng.standard_normal((k, B, J)))
        self._gemm(False, True, k, J, J, 1.0, buf["T"], ni, J, self.Msqrt, J, J * J,
                   0.0, buf["W"], ni, J, B)
        # G^T e, and the prior's (P U) z diag(dinv)^1/2
        self._gemm(True, False, N2, J, B, 1.0, self.G, N2, 0, buf["W"], J, ni, 0.0,
                   buf["ap"], J, nx, k)
        self._upload(buf["t1"], rng.standard_normal((k, N2, J)) * self.sqrt_dinv)
        self._gemm(False, False, N2, J, N2, 1.0, self.PU, N2, 0, buf["t1"], J, nx, 1.0,
                   buf["ap"], J, nx, k)
        out = numpy.empty((k, self.size))
        out[:, :nx] = self._download(buf["ap"], nx * k).reshape(k, nx)
        if not self.xi:
            return out
        # Xi: e, and the prior's E_c diag(lamS_c)^1/2 z Rm_c^T
        for g in self.groups:
            nb, o = g["nb"], g["start"]
            self._upload(buf["gt"], rng.standard_normal((k, nb, J)) * g["sqrt_lam"][:, None])
            self._gemm(False, True, nb, J, J, 1.0, buf["gt"], J, nb * J, g["Rm"], J, 0, 0.0,
                       buf["zi"], J, nb * J, k)
            self._gemm(False, False, nb, J, nb, 1.0, g["E"], nb, 0, buf["zi"], J, nb * J, 1.0,
                       (buf["W"], o * J), J, ni, k)
        Xi = numpy.empty((k, B, J))
        Xi[:, self.perm] = self._download(buf["W"], ni * k).reshape(k, B, J)
        out[:, nx:] = Xi.reshape(k, ni)
        return out


    # implementation details
    def _apply(self, buf, k):
        """
        (ap, api) = A (p, pi), in the interleaved coordinates of the preconditioner
        """
        J, B, N2 = self.J, self.B, self.N2
        kJ = k * J
        # T = G U p (+ E_c pi_c)
        self._gemm(False, False, B, kJ, N2, 1.0, self.GU, N2, 0, buf["p"], kJ, 0, 0.0,
                   buf["T"], kJ, 0, 1)
        for g in self.groups:
            nb, o = g["nb"], g["start"] * kJ
            self._gemm(False, False, nb, kJ, nb, 1.0, g["E"], nb, 0, (buf["pi"], o), kJ, 0,
                       1.0, (buf["T"], o), kJ, 0, 1)
        # W_b = T_b (beta M_b), T_b the (k x J) cells of block b
        self._gemm(False, False, k, J, J, 1.0, buf["T"], J, kJ, self.M, J, J * J, 0.0,
                   buf["W"], J, kJ, B)
        # ap = (G U)^T W + p diag(dinv)
        self._gemm(True, False, N2, kJ, B, 1.0, self.GU, N2, 0, buf["W"], kJ, 0, 0.0,
                   buf["ap"], kJ, 0, 1)
        self._launch("add_scaled_columns", N2 * kJ, buf["ap"], buf["p"], self.dinv,
                     numpy.int32(J), numpy.uint64(N2 * kJ))
        # api_c = E_c^T W_c + diag(lamS_c) pi_c Pm_c
        for g in self.groups:
            nb, o = g["nb"], g["start"] * kJ
            self._gemm(True, False, nb, kJ, nb, 1.0, g["E"], nb, 0, (buf["W"], o), kJ, 0, 0.0,
                       (buf["api"], o), kJ, 0, 1)
            self._gemm(False, False, nb * k, J, J, 1.0, (buf["pi"], o), J, 0, g["Pm"], J, 0, 0.0,
                       buf["gt"], J, 0, 1)
            self._launch("add_scaled_rows", nb * kJ, (buf["api"], o), buf["gt"], g["lam"],
                         numpy.int32(J), numpy.int32(k), numpy.uint64(nb * kJ))
        return


    def _precondition(self, buf, k):
        """
        z = (r V / denom) V^T; zi_c = (ri_c V_c / denom_c) V_c^T
        """
        J, N2 = self.J, self.N2
        kJ = k * J
        self._gemm(False, False, N2 * k, J, J, 1.0, buf["r"], J, 0, self.V, J, 0, 0.0,
                   buf["t1"], J, 0, 1)
        self._launch("divide", N2 * kJ, buf["t1"], self.denom, numpy.int32(J), numpy.int32(k),
                     numpy.uint64(N2 * kJ))
        self._gemm(False, True, N2 * k, J, J, 1.0, buf["t1"], J, 0, self.V, J, 0, 0.0,
                   buf["z"], J, 0, 1)
        for g in self.groups:
            nb, o = g["nb"], g["start"] * kJ
            self._gemm(False, False, nb * k, J, J, 1.0, (buf["ri"], o), J, 0, g["V"], J, 0, 0.0,
                       buf["gt"], J, 0, 1)
            self._launch("divide", nb * kJ, buf["gt"], g["denom"], numpy.int32(J),
                         numpy.int32(k), numpy.uint64(nb * kJ))
            self._gemm(False, True, nb * k, J, J, 1.0, buf["gt"], J, 0, g["V"], J, 0, 0.0,
                       (buf["zi"], o), J, 0, 1)
        return


    def _dot(self, buf, a, b, ai, bi, out, k):
        """
        {out}[s] = a_s . b_s + ai_s . bi_s
        """
        J = numpy.int32(self.J)
        self._launch("dot_rows", None, buf[a], buf[b], numpy.int32(self.N2), numpy.int32(k), J,
                     buf[out], numpy.int32(0), rows=k)
        if self.xi:
            self._launch("dot_rows", None, buf[ai], buf[bi], numpy.int32(self.B), numpy.int32(k),
                         J, buf[out], numpy.int32(1), rows=k)
        return


    def _buffers(self, k):
        """
        The work arrays for {k} right-hand sides, allocated when {k} changes
        """
        if self.buffers.get("k") == k:
            return self.buffers
        self.buffers = {}
        nx, ni = self.N2 * self.J, self.B * self.J
        gmax = max([g["nb"] for g in self.groups] + [1]) * self.J
        vector = altar.cuda.vector
        buf = {"k": k}
        for name in ("x", "r", "z", "p", "ap", "t1"):
            buf[name] = vector(shape=nx * k)
        for name in ("T", "W"):
            buf[name] = vector(shape=ni * k)
        for name in ("xi", "ri", "zi", "pi", "api"):
            buf[name] = vector(shape=(ni if self.xi else 1) * k)
        buf["gt"] = vector(shape=gmax * k)
        for name in ("rz", "rznew", "pap", "alpha", "beta"):
            buf[name] = vector(shape=k)
        self.buffers = buf
        return buf


    def _gemm(self, ta, tb, m, n, k, alpha, A, lda, sA, B, ldb, sB, beta, C, ldc, sC, batch):
        """
        Row major: C_i (m x n) = alpha op(A_i) op(B_i) + beta C_i, as the column major
        C^T = op(B)^T op(A)^T; the arrays are {altar.cuda} arrays or (array, offset) pairs
        """
        cublasx.gemm_strided_batched(
            altar.cuda.cublas_handle(), "float64",
            cublasx.OP_T if tb else cublasx.OP_N, cublasx.OP_T if ta else cublasx.OP_N,
            n, m, k, alpha, self._address(B), ldb, sB, self._address(A), lda, sA,
            beta, self._address(C), ldc, sC, batch)
        return


    def _launch(self, name, total, *args, rows=None):
        """
        Launch the kernel {name} over {total} cells, or one block per right-hand side with
        {rows}; the arrays among {args} go as their device addresses
        """
        kernel = self.kernels[f"{name}<double>"]
        arguments = [numpy.uintp(self._address(a)) if self._is_array(a) else a for a in args]
        if rows is not None:
            nvrtc.launch(kernel, rows, THREADS, *arguments, shared=THREADS * 8)
        else:
            blocks = max(1, min(nvrtc.blocks(total, THREADS), 8192))
            nvrtc.launch(kernel, blocks, THREADS, *arguments)
        return


    @staticmethod
    def _is_array(a):
        return isinstance(a, tuple) or hasattr(a, "grid")


    @staticmethod
    def _address(a):
        array, offset = a if isinstance(a, tuple) else (a, 0)
        return cublasx.address(array, offset, "float64")


    @staticmethod
    def _upload(array, values):
        altar.cuda.synchronize()
        values = numpy.ascontiguousarray(values, dtype=float).ravel()
        numpy.asarray(array)[:values.size] = values
        return


    @staticmethod
    def _download(array, count):
        altar.cuda.synchronize()
        return numpy.asarray(array)[:count].copy()


# end of file
