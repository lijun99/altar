# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# the slips of a kinematic model integrated out under a gaussian prior, for static displacements
# binned in time: the cumulative slip fractions of the patches, and the marginal likelihood

# externals
import concurrent.futures
import math
import os
import numpy
import scipy.linalg
import scipy.special


def cpu_threads():
    """
    The threads of the cpu likelihoods: OMP_NUM_THREADS, or the cpus this process may use
    """
    count = os.environ.get("OMP_NUM_THREADS")
    return max(1, int(count)) if count else len(os.sched_getaffinity(0))


def single_blas():
    """
    Whether the blas runs in one thread: OPENBLAS_NUM_THREADS or MKL_NUM_THREADS set to 1 at launch
    """
    return any(os.environ.get(name) == "1" for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"))


def over_samples(f, n, count):
    """
    The rows f(slice) of the slices of range(n) over {count} threads, if the blas runs in one
    thread; otherwise in one call, the blas threaded
    """
    count = min(count, n) if single_blas() else 1
    if count <= 1:
        return f(slice(0, n))
    edges = numpy.linspace(0, n, count + 1).astype(int)
    with concurrent.futures.ThreadPoolExecutor(count) as pool:
        parts = list(pool.map(lambda i: f(slice(edges[i], edges[i + 1])), range(count)))
    return numpy.concatenate(parts)


def cumulative(s, Tr, stf="triangle", peak=0.5):
    """
    V, the slip of a point {s} after its onset, a fraction of its slip, and W, the integral of V,
    for a source time function of unit area over the rise time {Tr}: a triangle peaking at {peak}
    of it, or a raised cosine; as stf_cumulative in stf.h
    """
    s, Tr = numpy.broadcast_arrays(numpy.asarray(s, dtype=float), numpy.asarray(Tr, dtype=float))
    V, W = numpy.zeros(s.shape), numpy.zeros(s.shape)
    late = s >= Tr
    if stf == "cosine":
        on = (s > 0) & ~late
        x = 2 * math.pi * s[on] / Tr[on]
        V[on] = s[on] / Tr[on] - numpy.sin(x) / (2 * math.pi)
        W[on] = s[on] ** 2 / (2 * Tr[on]) + Tr[on] * (numpy.cos(x) - 1) / (4 * math.pi ** 2)
        V[late] = 1
        W[late] = s[late] - Tr[late] / 2
        return V, W
    rising = (s > 0) & (s < peak * Tr)
    V[rising] = s[rising] ** 2 / (peak * Tr[rising] ** 2)
    W[rising] = s[rising] * V[rising] / 3
    falling = s >= peak * Tr
    r = numpy.where(late, 0.0, Tr - s)[falling]
    q = (1 - peak) * Tr[falling] ** 2
    V[falling] = 1 - r * r / q
    W[falling] = s[falling] - (1 + peak) * Tr[falling] / 3 + r ** 3 / (3 * q)
    return V, W


def fractions(onsets, risetime, dt, bins, stf="triangle", peak=0.5, integration="bin"):
    """
    The cumulative slip fraction of each patch in each bin [k dt, (k+1) dt), (samples, patches,
    bins), the mean over its source points: with "bin", its average over the bin; with "step", at
    the end of the bin; {onsets}, (samples, patches, points), the arrival times at the source
    points, and {risetime}, (samples, patches)
    """
    edges = dt * numpy.arange(bins + 1)
    s = edges[None, None, None, :] - onsets[..., None]
    V, W = cumulative(s, risetime[:, :, None, None], stf, peak)
    if integration == "step":
        return V[..., 1:].mean(axis=2)
    if integration == "bin":
        return (numpy.diff(W, axis=-1) / dt).mean(axis=2)
    raise ValueError(f"integration: step or bin; got '{integration}'")


def box(L, mean, uniforms, paths=False):
    """
    log Pr(x_k >= 0 for the last k components), x ~ N({mean}, P^-1), P = L L^T, for each of a
    batch, by GHK with the {uniforms}, (paths, k): with y = x - mean, L^T y = z standard normal,
    so from the last row up each z is drawn above the bound its row sets, and each path weighs
    the product of the probabilities of the bounds; with {paths}, also y and the log weights
    """
    B, n = mean.shape
    count, k = uniforms.shape
    y = numpy.zeros((B, count, k))
    logw = numpy.zeros((B, count))
    for i in range(k - 1, -1, -1):
        g = n - k + i
        s = numpy.matmul(y[:, :, i + 1:], L[:, g + 1:, g, None])[..., 0]
        Lgg = L[:, g, g][:, None]
        # x_g >= 0 when z_g >= s - L_gg mean_g
        lp = scipy.special.log_ndtr(Lgg * mean[:, g][:, None] - s)
        logw += lp
        z = -scipy.special.ndtri_exp(numpy.log1p(-uniforms[None, :, i]) + lp)
        y[:, :, i] = (z - s) / Lgg
    log = scipy.special.logsumexp(logw, axis=1) - math.log(count)
    return (log, y, logw) if paths else log


def sobol(count, k, seed):
    """
    {count} scrambled sobol points in [0, 1)^k
    """
    from scipy.stats import qmc
    return qmc.Sobol(d=k, scramble=True, seed=seed).random(count)


class Static:
    """
    Static displacements after all the slip, d_s = G_s m + R r + e, e ~ N(0, C_s), with the slips
    m, the ramps r, the last {ramps} columns of {G}, (observations, slips + ramps), and C_s {C}, a
    variance or a matrix; the ramps integrated out under a flat prior, by projecting them out of
    the whitened system. Their parts of the marginal likelihood of {GaussianSlip}: P += G^T W G,
    b += G^T W d, and the terms d^T W d + log|C_s| + log|R^T C_s^-1 R| + (observations - ramps)
    log 2 pi, with W = C_s^-1 - C_s^-1 R (R^T C_s^-1 R)^-1 R^T C_s^-1
    """


    def __init__(self, G, data, C, ramps=0):
        G = numpy.asarray(G, dtype=float)
        d = numpy.asarray(data, dtype=float).ravel()
        if G.ndim != 2 or G.shape[0] != d.size or G.shape[1] <= ramps:
            raise ValueError(f"static green's functions ({d.size} observations, slips + {ramps} ramps); got {G.shape}")
        self.G, self.ramps, self.slips = G, ramps, G.shape[1] - ramps
        if numpy.ndim(C) == 0:
            sd = math.sqrt(float(C))
            whiten = lambda x: x / sd
            logdet = d.size * math.log(float(C))
        else:
            L = numpy.linalg.cholesky(numpy.asarray(C, dtype=float))
            whiten = lambda x: scipy.linalg.solve_triangular(L, x, lower=True)
            logdet = 2 * numpy.log(numpy.diag(L)).sum()
        Gw, dw = whiten(G), whiten(d)
        Gm, dm = Gw[:, :self.slips], dw
        if ramps:
            Q, R = numpy.linalg.qr(Gw[:, self.slips:])
            Gm, dm = Gm - Q @ (Q.T @ Gm), dm - Q @ (Q.T @ dm)
            logdet += 2 * numpy.log(numpy.abs(numpy.diag(R))).sum()
            self._Q, self._R, self._Gw, self._dw = Q, R, Gw[:, :self.slips], dw
        self.P, self.b = Gm.T @ Gm, Gm.T @ dm
        self.terms = dm @ dm + logdet + (d.size - ramps) * math.log(2 * math.pi)
        self.observations = d.size


    def ramp(self, slip):
        """
        The ramps that fit the data best given the slips {slip}, (..., slips)
        """
        slip = numpy.asarray(slip, dtype=float)
        if not self.ramps:
            return numpy.zeros(slip.shape[:-1] + (0,))
        residual = self._dw - slip @ self._Gw.T
        return numpy.linalg.solve(self._R, (residual @ self._Q)[..., None])[..., 0]


    def predict(self, slip):
        """
        The static data of the slips {slip}, (..., slips), with the ramps that fit them best
        """
        slip = numpy.asarray(slip, dtype=float)
        return slip @ self.G[:, :self.slips].T + self.ramp(slip) @ self.G[:, self.slips:].T


class GaussianSlip:
    """
    The data d, per component c, (stations, bins), linear in the slips m of the patches along two
    rakes, d_c = sum_j G_c[j] m_j F[p(j)], with the static green's functions {G}, (2 patches,
    components x stations), and F the cumulative slip fraction of each patch in each bin; the noise
    of covariance blockdiag_c(S_c ⊗ R_c), an {altar.data.Kronecker}; and the gaussian prior
    m ~ N(0, sigma^2 C0), C0 the identity unless given. Then slip integrates out:

        P = C0^-1 / sigma^2 + sum_c (G_c S_c^-1 G_c^T) o (F R_c^-1 F^T)
        b = sum_c rowsum(F o (G_c S_c^-1 D_c R_c^-1))
        log p(d) = -(d^T Cd^-1 d - b^T P^-1 b + log|P| + log|sigma^2 C0| + log|Cd| + n log 2 pi) / 2

    and m | d ~ N(P^-1 b, P^-1); F tiled over the two rakes

    With {positive}, a mask of the slips, those are bounded below by 0, the prior truncated:
    log p(d) gains log Pr(m_S >= 0 | d) - log Pr(m_S >= 0), the first by GHK with {paths} fixed
    sobol points, the same for every sample, so that the likelihood is a smooth function of the
    fronts; the second exact for C0 the identity, else by GHK with many points

    With {static}, a {Static}, static data of the same slips add constant terms to P, b and
    log p(d), which is then the likelihood of both data sets
    """


    def __init__(self, G, data, noise, C0=None, positive=None, paths=256, seed=0, static=None):
        self.noise = noise
        C, ns, nt = noise.components, noise.stations, noise.bins
        G = numpy.asarray(G, dtype=float)
        if G.shape[1] != C * ns or G.shape[0] % 2:
            raise ValueError(f"green's functions (2 patches, {C} x {ns}); got {G.shape}")
        self.G = G.reshape(G.shape[0], C, ns)
        self.slips = G.shape[0]
        self.patches = self.slips // 2
        D = numpy.asarray(data, dtype=float).reshape(C, ns, nt)
        self.D = D
        solve = scipy.linalg.solve_triangular
        self.GG, self.GSD, self.Rinv = [], [], []
        quadratic = 0.0
        for c in range(C):
            LS, LR = noise.LS[c], noise.LR[c]
            # G S^-1 G^T = Gw Gw^T, Gw = G L_S^-T
            Gw = solve(LS, self.G[:, c].T, lower=True).T
            # L_S^-1 D L_R^-T
            Dw = solve(LR, solve(LS, D[c], lower=True).T, lower=True).T
            quadratic += (Dw * Dw).sum()
            LRinv = solve(LR, numpy.eye(nt), lower=True)
            self.GG.append(Gw @ Gw.T)
            # G S^-1 D R^-1 = Gw (L_S^-1 D L_R^-T) L_R^-1
            self.GSD.append(Gw @ Dw @ LRinv)
            self.Rinv.append(LRinv.T @ LRinv)
        self.quadratic = quadratic
        self.observations = C * ns * nt
        # the static data, constant parts of the system
        if static is not None and static.slips != self.slips:
            raise ValueError(f"static data of {static.slips} slips, not {self.slips}")
        self.static = static
        self.Ps = numpy.zeros((self.slips, self.slips)) if static is None else static.P
        self.bs = numpy.zeros(self.slips) if static is None else static.b
        self.static_terms = 0.0 if static is None else static.terms
        if C0 is None:
            self.C0inv, self.logdetC0 = numpy.eye(self.slips), 0.0
        else:
            C0 = numpy.asarray(C0, dtype=float)
            L0 = numpy.linalg.cholesky(C0)
            self.C0inv = scipy.linalg.cho_solve((L0, True), numpy.eye(self.slips))
            self.logdetC0 = 2 * numpy.log(numpy.diag(L0)).sum()
        # the bounded slips, last in the order of the system
        self.positive = numpy.zeros(self.slips, bool) if positive is None else numpy.asarray(positive, bool)
        self.bounded = int(self.positive.sum())
        self.order = numpy.concatenate([numpy.flatnonzero(~self.positive), numpy.flatnonzero(self.positive)])
        self.logZ0 = 0.0
        if self.bounded:
            self.uniforms = sobol(paths, self.bounded, seed)
            if C0 is None:
                self.logZ0 = -self.bounded * math.log(2)
            else:
                L0 = numpy.linalg.cholesky(self.C0inv[numpy.ix_(self.order, self.order)])
                self.logZ0 = float(box(L0[None], numpy.zeros((1, self.slips)), sobol(2 ** 14, self.bounded, seed))[0])
        return


    def system(self, F, sigma):
        """
        P and b for the fractions {F}, (samples, patches, bins), and the prior scales {sigma},
        (samples,)
        """
        F = numpy.asarray(F, dtype=float)
        n, Np = F.shape[0], self.patches
        sigma = numpy.broadcast_to(numpy.asarray(sigma, dtype=float), (n,))
        P = self.C0inv[None] / (sigma ** 2)[:, None, None] + self.Ps[None]
        b = numpy.repeat(self.bs[None], n, axis=0)
        work = numpy.empty((n, Np, Np))
        blocks = (slice(0, Np), slice(Np, None))
        for GG, GSD, Rinv in zip(self.GG, self.GSD, self.Rinv):
            FRF = (F @ Rinv) @ numpy.swapaxes(F, 1, 2)
            for i in blocks:
                for j in blocks:
                    P[:, i, j] += numpy.multiply(GG[None, i, j], FRF, out=work)
            b[:, :Np] += numpy.einsum("spk,pk->sp", F, GSD[:Np])
            b[:, Np:] += numpy.einsum("spk,pk->sp", F, GSD[Np:])
        return P, b


    def factor(self, F, sigma):
        """
        L, P = L L^T, z = L^-1 b and the mean P^-1 b, in the order of the system, the bounded
        slips last
        """
        P, b = self.system(F, sigma)
        if self.bounded:
            P, b = P[:, self.order][:, :, self.order], b[:, self.order]
        L = numpy.linalg.cholesky(P)
        solve = scipy.linalg.solve_triangular
        z = numpy.stack([solve(l, v, lower=True, check_finite=False) for l, v in zip(L, b)])
        mean = numpy.stack([solve(l, v, lower=True, trans="T", check_finite=False) for l, v in zip(L, z)])
        return P, L, z, mean


    def loglikelihood(self, F, sigma, threads=None):
        """
        The marginal log likelihood of the data, (samples,), the samples over {threads}, by
        default those of {cpu_threads()}
        """
        F = numpy.asarray(F, dtype=float)
        sigma = numpy.broadcast_to(numpy.asarray(sigma, dtype=float), (F.shape[0],))
        count = cpu_threads() if threads is None else threads
        return over_samples(lambda rows: self._loglikelihood(F[rows], sigma[rows]), F.shape[0], count)


    def _loglikelihood(self, F, sigma):
        """
        The marginal log likelihood of the data, (samples,), in this thread
        """
        P, L, z, mean = self.factor(F, sigma)
        logdetP = 2 * numpy.log(numpy.diagonal(L, axis1=1, axis2=2)).sum(axis=1)
        sigma = numpy.broadcast_to(numpy.asarray(sigma, dtype=float), (P.shape[0],))
        logdetCm = self.slips * numpy.log(sigma ** 2) + self.logdetC0
        log = -0.5 * (self.quadratic - (z * z).sum(axis=1) + logdetP + logdetCm + self.noise.logdet
                      + self.observations * math.log(2 * math.pi) + self.static_terms)
        if self.bounded:
            log += box(L, mean, self.uniforms) - self.logZ0
        return log


    def posterior(self, F, sigma):
        """
        The mean of the slips given the data, (samples, 2 patches), and L, P = L L^T, the factor of
        their precision
        """
        P, b = self.system(F, sigma)
        L = numpy.linalg.cholesky(P)
        mean = numpy.linalg.solve(P, b[..., None])[..., 0]
        return mean, L


    def draw(self, F, sigma, rng, count=1, paths=1024):
        """
        {count} draws of the slips given the data for each sample, (samples, count, 2 patches);
        with bounded slips, by resampling {paths} GHK paths by their weights, then the others
        given those
        """
        if self.bounded:
            return self._truncated(F, sigma, rng, count, paths)
        mean, L = self.posterior(F, sigma)
        z = rng.standard_normal((mean.shape[0], self.slips, count))
        # m = mean + L^-T z
        x = numpy.stack([scipy.linalg.solve_triangular(l, zi, lower=True, trans="T") for l, zi in zip(L, z)])
        return mean[:, None, :] + numpy.moveaxis(x, 2, 1)


    def mean(self, F, sigma, rng, count=256):
        """
        The mean of the slips given the data, (samples, 2 patches); with bounded slips, of {count}
        draws
        """
        if not self.bounded:
            return self.posterior(F, sigma)[0]
        return self.draw(F, sigma, rng, count).mean(axis=1)


    def _truncated(self, F, sigma, rng, count, paths):
        """
        Draws of the slips, the bounded ones by sampling-importance-resampling of GHK paths, the
        free ones from their gaussian given those
        """
        P, L, z, mean = self.factor(F, sigma)
        k, f = self.bounded, self.slips - self.bounded
        out = numpy.empty((P.shape[0], count, self.slips))
        for s in range(P.shape[0]):
            _, y, logw = box(L[s:s + 1], mean[s:s + 1], rng.random((paths, k)), paths=True)
            w = numpy.exp(logw[0] - logw[0].max())
            yS = y[0, rng.choice(paths, count, p=w / w.sum())]
            x = numpy.empty((count, self.slips))
            x[:, f:] = yS
            if f:
                Lff = L[s, :f, :f]
                shift = scipy.linalg.cho_solve((Lff, True), P[s, :f, f:] @ yS.T)
                noise = scipy.linalg.solve_triangular(Lff, rng.standard_normal((f, count)), lower=True, trans="T")
                x[:, :f] = (noise - shift).T
            x += mean[s]
            out[s][:, self.order] = x
        return out


    def predict(self, F, slip):
        """
        The data, (samples, components x stations x bins), of the slips {slip}, (samples,
        2 patches), and the fractions {F}, (samples, patches, bins)
        """
        F2 = numpy.concatenate([F, F], axis=1)
        d = numpy.einsum("jcs,nj,njk->ncsk", self.G, numpy.asarray(slip, dtype=float), F2)
        return d.reshape(d.shape[0], -1)


# end of file
