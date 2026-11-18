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
import math
import numpy
import scipy.linalg


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
    """


    def __init__(self, G, data, noise, C0=None):
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
        if C0 is None:
            self.C0inv, self.logdetC0 = numpy.eye(self.slips), 0.0
        else:
            C0 = numpy.asarray(C0, dtype=float)
            L0 = numpy.linalg.cholesky(C0)
            self.C0inv = scipy.linalg.cho_solve((L0, True), numpy.eye(self.slips))
            self.logdetC0 = 2 * numpy.log(numpy.diag(L0)).sum()
        return


    def system(self, F, sigma):
        """
        P and b for the fractions {F}, (samples, patches, bins), and the prior scales {sigma},
        (samples,)
        """
        F = numpy.asarray(F, dtype=float)
        n, Np = F.shape[0], self.patches
        sigma = numpy.broadcast_to(numpy.asarray(sigma, dtype=float), (n,))
        P = self.C0inv[None] / (sigma ** 2)[:, None, None]
        b = numpy.zeros((n, self.slips))
        for GG, GSD, Rinv in zip(self.GG, self.GSD, self.Rinv):
            FRF = numpy.einsum("spk,kl,sql->spq", F, Rinv, F, optimize=True)
            P = P + GG[None] * numpy.tile(FRF, (1, 2, 2))
            b[:, :Np] += numpy.einsum("spk,pk->sp", F, GSD[:Np])
            b[:, Np:] += numpy.einsum("spk,pk->sp", F, GSD[Np:])
        return P, b


    def loglikelihood(self, F, sigma):
        """
        The marginal log likelihood of the data, (samples,)
        """
        P, b = self.system(F, sigma)
        L = numpy.linalg.cholesky(P)
        z = numpy.linalg.solve(L, b[..., None])[..., 0]
        logdetP = 2 * numpy.log(numpy.diagonal(L, axis1=1, axis2=2)).sum(axis=1)
        sigma = numpy.broadcast_to(numpy.asarray(sigma, dtype=float), (P.shape[0],))
        logdetCm = self.slips * numpy.log(sigma ** 2) + self.logdetC0
        return -0.5 * (self.quadratic - (z * z).sum(axis=1) + logdetP + logdetCm + self.noise.logdet
                       + self.observations * math.log(2 * math.pi))


    def posterior(self, F, sigma):
        """
        The mean of the slips given the data, (samples, 2 patches), and L, P = L L^T, the factor of
        their precision
        """
        P, b = self.system(F, sigma)
        L = numpy.linalg.cholesky(P)
        mean = numpy.linalg.solve(P, b[..., None])[..., 0]
        return mean, L


    def draw(self, F, sigma, rng, count=1):
        """
        {count} draws of the slips given the data for each sample, (samples, count, 2 patches)
        """
        mean, L = self.posterior(F, sigma)
        z = rng.standard_normal((mean.shape[0], self.slips, count))
        # m = mean + L^-T z
        x = numpy.stack([scipy.linalg.solve_triangular(l, zi, lower=True, trans="T") for l, zi in zip(L, z)])
        return mean[:, None, :] + numpy.moveaxis(x, 2, 1)


    def predict(self, F, slip):
        """
        The data, (samples, components x stations x bins), of the slips {slip}, (samples,
        2 patches), and the fractions {F}, (samples, patches, bins)
        """
        F2 = numpy.concatenate([F, F], axis=1)
        d = numpy.einsum("jcs,nj,njk->ncsk", self.G, numpy.asarray(slip, dtype=float), F2)
        return d.reshape(d.shape[0], -1)


# end of file
