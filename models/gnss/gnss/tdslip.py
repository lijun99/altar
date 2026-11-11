# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

"""
The numerics of time-dependent slip on a fault mesh from the trajectories of a GNSS case: the
linear-gaussian posterior of the slip X, (2N x J), on N patches (dip slip first, then strike
slip) and J temporal basis functions shared by all the blocks of the case, and, optionally, of a
transient noise Xi, (B x J), correlated between the stations

Each block b (a station and a component) of the case is a sum of groups of basis functions (see
{altar.models.gnss.design}). The groups that are the temporal basis of the slip, typically the
I-splines, have the coefficients theta_b = (G X + Xi)_b, with G (B x 2N) the elastic Green's
functions of the block's station and component; the other groups (offset, trend, seasonal,
steps, ...) are nuisance with gaussian priors, integrated out exactly per block. The noise of a
block is white (its uncertainties, scaled) plus exponentially correlated processes, or white
only. With the nuisance integrated out, the data enter through, per block,

    log p(d_b | theta_b) = -1/2 theta_b^T M_b theta_b + h_b^T theta_b + c_b.

The prior of X is gaussian and separable, P (x) D: P = blockdiag(C^-1, C^-1 / rho^2) in space,
C = exp(-d / length) between patch centroids, and D = diag(1 / sigma_j^2) in time, sigma_j =
sigma0 (s_j / scale_ref)^power for a column of scale s_j. The optional noise Xi has, per
component c, the matrix-normal prior N(0, S_c, c0_c (Mbar_c + I / cap^2)^-1), S_c = exp(-d / ell_c)
between the stations, Mbar_c the mean M_b of the component; the independent noise is then scaled
by 1 / (1 - c0_c). The posterior at beta is N(A^-1 b, A^-1), A of size 2NJ (+ BJ), applied
matrix-free; {Operator} is the {LinearOperator} of the perturbation sampler.
"""

from __future__ import annotations
import json
import math
import numpy
from scipy.linalg import cho_factor, cho_solve, eigh, solve_triangular

COMPONENTS = ("east", "north", "up")
KM_PER_DEG = 111.195


def load_noise(path, design):
    """
    The correlated noise of the blocks: {"taus_days": [...], "blocks": [{"station", "component",
    "w", "amps_mm": [...]}, or null, ...]}, one entry per block of {design}, in its order; the
    covariance of block b over its valid days is w^2 sigma^2 + sum_m a_m^2 exp(-|dt| / tau_m).
    Blocks without parameters (null) get their component's medians. Returns (taus, rows, filled)
    """
    model = json.loads(open(path).read())
    rows = model["blocks"]
    if len(rows) != design.blocks:
        raise ValueError(f"{path}: {len(rows)} blocks, the case has {design.blocks}")
    for b, r in enumerate(rows):
        if r and (r["station"], r["component"]) != (design.station[b], design.component[b]):
            raise ValueError(f"{path}: block {b} is {r['station']} {r['component']}, the case's "
                             f"is {design.station[b]} {design.component[b]}")
    taus = [float(t) for t in model["taus_days"]]
    med = {}
    for c in set(design.component):
        rc = [r for r in rows if r and r["component"] == c]
        if rc:
            med[c] = {"w": float(numpy.median([r["w"] for r in rc])),
                      "amps_mm": [float(numpy.median([r["amps_mm"][m] for r in rc]))
                                  for m in range(len(taus))]}
    filled = [b for b, r in enumerate(rows) if r is None]
    rows = [r if r else med[design.component[b]] for b, r in enumerate(rows)]
    return taus, rows, filled


def block_covariance(sigma, t, row, taus):
    """
    The noise covariance of one block over its valid days {t}, uncertainties {sigma}
    """
    K = numpy.diag((row["w"] * sigma) ** 2)
    if taus:
        dt = numpy.abs(t[:, None] - t[None, :])
        for a, tau in zip(row["amps_mm"], taus):
            K = K + a ** 2 * numpy.exp(-dt / tau)
    return K


def load_greens(path, design):
    """
    The Green's functions of the blocks of {design}, (B x 2N), dip slip then strike slip, and the
    patch centroids (N x 3, km), from {path}: "stations" (S names), "gf" (S x 3 components east,
    north, up x 2 rakes, strike slip then dip slip, x N patches, displacement per unit slip) and
    "centroids_km" (N x 3, in any cartesian frame)
    """
    import h5py
    with h5py.File(path, "r") as h:
        names = [x.decode() if isinstance(x, bytes) else str(x) for x in h["stations"][:]]
        gf = h["gf"][:]
        centroids = h["centroids_km"][:]
    where = {n: i for i, n in enumerate(names)}
    missing = sorted({s for s in design.station if s not in where})
    if missing:
        raise ValueError(f"{path}: no Green's functions for the stations {missing}")
    rows = [gf[where[s], COMPONENTS.index(c)] for s, c in zip(design.station, design.component)]
    return numpy.stack([numpy.concatenate([r[1], r[0]]) for r in rows]), centroids


def station_distances(lat, lon):
    """
    Great-circle distances (km) between the stations of the blocks
    """
    p, l = numpy.radians(lat), numpy.radians(lon)
    c = (numpy.sin(p[:, None]) * numpy.sin(p[None, :])
         + numpy.cos(p[:, None]) * numpy.cos(p[None, :]) * numpy.cos(l[:, None] - l[None, :]))
    return 6371.0 * numpy.arccos(numpy.clip(c, -1, 1))


class Slip:
    """
    The posterior of the slip (and the correlated noise) of a GNSS case; see the module
    """

    def __init__(self, design, slip_groups, nuisance_sigma, greens, centroids, noise=None,
                 sigma0=10.0, scale_ref=16.0, power=0.5, length=30.0, rho=0.3, spatial=None,
                 cap=10.0):
        d = self.design = design
        B, n = d.blocks, d.days
        names = [g.name for g in d.groups]
        unknown = [s for s in slip_groups if s not in names]
        if unknown:
            raise ValueError(f"slip groups {unknown} aren't groups of the case {names}")
        if not slip_groups:
            raise ValueError("no slip groups: name the case's groups that are the slip's "
                             "temporal basis")
        for g in d.groups:
            if g.name in slip_groups and not g.shared:
                raise ValueError(f"group '{g.name}' is per block; the slip's temporal basis must "
                                 f"be shared by all the blocks")
        # the columns of every block: the groups in the case's order
        columns, c = {}, 0
        for g in d.groups:
            columns[g.name], c = slice(c, c + g.width), c + g.width
        width = c
        tcols = numpy.concatenate([numpy.arange(width)[columns[s]] for s in slip_groups])
        ncols = numpy.setdiff1d(numpy.arange(width), tcols)
        self.slip_groups, self.tcols, self.ncols = list(slip_groups), tcols, ncols
        # the nuisance priors
        fixed = numpy.zeros(width)
        for g in d.groups:
            if g.name in slip_groups:
                continue
            sigma = nuisance_sigma.get(g.name)
            if sigma is None:
                sigma = next((v for k, v in nuisance_sigma.items()
                              if g.name.startswith(k) or g.kind == k), None)
            if sigma is None or sigma <= 0:
                raise ValueError(f"no gaussian prior sigma for the nuisance group '{g.name}' "
                                 f"({g.kind})")
            fixed[columns[g.name]] = 1 / float(sigma) ** 2
        # per block: the normal equations under the noise, the nuisance integrated out
        J = tcols.size
        data = d.data.reshape(B, n)
        t = numpy.asarray(d.time, dtype=float)
        taus, rows = (noise if noise is not None else ([], None))
        self.M = numpy.empty((B, J, J))
        self.h = numpy.empty((B, J))
        self.c = numpy.empty(B)
        for b in range(B):
            X = numpy.hstack([g.G if g.shared else g.G[b] for g in d.groups])
            ok = d.weight[b] > 0
            sig = 1 / d.weight[b][ok]
            y = data[b][ok] * sig                                       # mm
            row = rows[b] if rows is not None else {"w": 1.0, "amps_mm": []}
            K = block_covariance(sig, t[ok], row, taus)
            L = numpy.linalg.cholesky(K)
            Xw = solve_triangular(L, X[ok], lower=True)
            dw = solve_triangular(L, y, lower=True)
            XtX, Xtd = Xw.T @ Xw, Xw.T @ dw
            Ann = XtX[numpy.ix_(ncols, ncols)] + numpy.diag(fixed[ncols])
            Ant = XtX[numpy.ix_(ncols, tcols)]
            La = numpy.linalg.cholesky(Ann)
            sol = cho_solve((La, True), numpy.column_stack([Ant, Xtd[ncols]]))
            M = XtX[numpy.ix_(tcols, tcols)] - Ant.T @ sol[:, :-1]
            self.M[b] = 0.5 * (M + M.T)
            self.h[b] = Xtd[tcols] - Ant.T @ sol[:, -1]
            # log p(d_b | theta_b) = -1/2 theta^T M theta + h^T theta + c
            self.c[b] = -0.5 * (dw @ dw - Xtd[ncols] @ sol[:, -1]
                                + 2 * numpy.log(numpy.diag(L)).sum()
                                + ok.sum() * math.log(2 * math.pi)
                                + 2 * numpy.log(numpy.diag(La)).sum()
                                - numpy.log(fixed[ncols]).sum())
        # the temporal prior
        scale = numpy.concatenate([
            numpy.asarray(g.columns.get("scale", numpy.full(g.width, scale_ref)), dtype=float)
            if isinstance(g.columns, dict) else numpy.full(g.width, scale_ref)
            for s in slip_groups for g in d.groups if g.name == s])
        self.sigma = sigma0 * (scale / scale_ref) ** power
        self.dinv = 1 / self.sigma ** 2
        # the spatial prior
        self.G = numpy.asarray(greens, dtype=float)
        if self.G.shape[0] != B or self.G.shape[1] % 2:
            raise ValueError(f"Green's functions {self.G.shape}, the case has {B} blocks")
        N = self.N = self.G.shape[1] // 2
        cen = numpy.asarray(centroids, dtype=float)
        C = numpy.exp(-numpy.linalg.norm(cen[:, None] - cen[None], axis=-1) / length)
        self.LC = numpy.linalg.cholesky(C)
        Ci = cho_solve((self.LC, True), numpy.identity(N))
        Ci = 0.5 * (Ci + Ci.T)
        P = numpy.zeros((2 * N, 2 * N))
        P[:N, :N], P[N:, N:] = Ci, Ci / rho ** 2
        self.P, self.rho = P, rho
        self.J = J
        # the log normalization of the prior of X, precision P (x) D: 1/2 log|P (x) D| - NJ log 2pi,
        # log|P (x) D| = J log|P| + 2N log|D|, log|P| = -2 log|C| - 2N log rho
        logdetC = 2 * numpy.log(numpy.diag(self.LC)).sum()
        self.prior_constant = (0.5 * (J * (-2 * logdetC - 2 * N * math.log(rho))
                                      - 2 * N * numpy.log(self.sigma ** 2).sum())
                               - N * J * math.log(2 * math.pi))
        # the noise correlated between the stations
        self.groups = []
        if spatial:
            comp = numpy.array(d.component)
            lacking = sorted(set(comp) - set(spatial))
            if lacking:
                raise ValueError(f"no correlated-noise parameters for the components {lacking}")
            fac = numpy.ones(B)
            Mbar = {}
            for cmp in COMPONENTS:
                idx = numpy.flatnonzero(comp == cmp)
                if idx.size:
                    Mbar[cmp] = self.M[idx].mean(axis=0)
                    fac[idx] = 1 / (1 - spatial[cmp]["c0"])
            self.M *= fac[:, None, None]
            self.h *= fac[:, None]
            lat, lon = numpy.asarray(d.lat), numpy.asarray(d.lon)
            for cmp in COMPONENTS:
                idx = numpy.flatnonzero(comp == cmp)
                if not idx.size:
                    continue
                c0, ell = float(spatial[cmp]["c0"]), float(spatial[cmp]["ell_km"])
                S = (numpy.exp(-station_distances(lat[idx], lon[idx]) / ell)
                     + 1e-9 * numpy.identity(idx.size))
                LS = numpy.linalg.cholesky(S)
                Sinv = cho_solve((LS, True), numpy.identity(idx.size))
                Sinv = 0.5 * (Sinv + Sinv.T)
                Pm = (Mbar[cmp] + numpy.identity(J) / cap ** 2) / c0
                Pm = 0.5 * (Pm + Pm.T)
                lam, Q = numpy.linalg.eigh(Pm)
                Rm = Q * numpy.sqrt(numpy.maximum(lam, 0))[None, :]
                lamS, E = numpy.linalg.eigh(Sinv)
                # 1/2 log|S^-1 (x) Pm| - (nb J / 2) log 2pi, log|S^-1| = -2 sum log diag LS
                const = (0.5 * (idx.size * numpy.log(numpy.maximum(lam, 1e-300)).sum()
                                - J * 2 * numpy.log(numpy.diag(LS)).sum())
                         - 0.5 * idx.size * J * math.log(2 * math.pi))
                self.groups.append({"component": cmp, "idx": idx, "c0": c0, "ell": ell, "LS": LS,
                                    "Sinv": Sinv, "Pm": Pm, "Rm": Rm, "E": E, "lamS": lamS,
                                    "constant": const})
        self.size = 2 * N * J + (B * J if self.groups else 0)
        self._operators = {}
        return


    # the layout of a sample: X (2N x J) row major, then Xi (B x J) row major
    def split(self, theta):
        """
        The slip (k x 2N x J) and the noise (k x B x J, or None) of the rows of {theta}
        """
        theta = numpy.atleast_2d(theta)
        k, nx = theta.shape[0], 2 * self.N * self.J
        X = theta[:, :nx].reshape(k, 2 * self.N, self.J)
        Xi = theta[:, nx:].reshape(k, self.design.blocks, self.J) if self.groups else None
        return X, Xi


    def coefficients(self, theta):
        """
        The coefficients of the slip groups of every block, (k x B x J): G X + Xi
        """
        X, Xi = self.split(theta)
        T = numpy.matmul(self.G, X)
        return T if Xi is None else T + Xi


    def log_likelihood(self, theta):
        """
        The log likelihood of the data, the nuisance integrated out, for each row of {theta}
        (with the noise correlated between stations, up to a constant)
        """
        T = self.coefficients(theta)
        q = numpy.einsum("kbi,bij,kbj->k", T, self.M, T)
        return -0.5 * q + numpy.einsum("kbj,bj->k", T, self.h) + self.c.sum()


    def log_prior(self, theta):
        """
        The log prior of each row of {theta}
        """
        X, Xi = self.split(theta)
        PX = numpy.matmul(self.P, X)
        out = -0.5 * numpy.einsum("kij,kij,j->k", X, PX, self.dinv) + self.prior_constant
        for g in self.groups:
            Z = Xi[:, g["idx"]]
            out += (-0.5 * numpy.einsum("kij,kij->k", numpy.matmul(g["Sinv"], Z),
                                        numpy.matmul(Z, g["Pm"])) + g["constant"])
        return out


    def operator(self, beta=1.0):
        """
        The {LinearOperator} of the posterior at {beta}
        """
        op = self._operators.get(beta)
        if op is None:
            op = self._operators[beta] = Operator(self, beta)
        return op


class Operator:
    """
    The posterior precision A of a {Slip} at beta, matrix-free, with a Kronecker preconditioner,
    for {altar.bayesian.samplers.Perturbation}: on (rows x size) arrays
    """

    bounded = False

    def __init__(self, slip, beta):
        s = self.slip = slip
        self.beta, self.size = beta, slip.size
        self.N, self.J, self.B = s.N, s.J, s.design.blocks
        bM = beta * s.M
        self.bM = bM
        # the preconditioner of the slip: every M_b replaced by w_b Mbar of the same trace;
        # K1 U = P U diag(mu), U^T P U = I; Mbar V = D V diag(nu), V^T D V = I
        tr = numpy.einsum("bii->b", bM)
        Mbar = bM.mean(axis=0)
        w = tr / max(numpy.trace(Mbar), 1e-300)
        K1 = s.G.T @ (w[:, None] * s.G)
        self.mu, self.U = eigh(K1, s.P)
        self.nu, self.V = eigh(Mbar, numpy.diag(s.dinv))
        self.denom = (numpy.maximum(self.mu, 0)[:, None] * numpy.maximum(self.nu, 0)[None, :]
                      + 1.0)
        # and of the noise: per component, the exact inverse of Xi Mt + S^-1 Xi Pm
        self.pre = []
        for g in s.groups:
            Mt = bM[g["idx"]].mean(axis=0)
            nu, V = eigh(0.5 * (Mt + Mt.T), g["Pm"])
            self.pre.append({"E": g["E"], "V": V,
                             "denom": numpy.maximum(g["lamS"], 0)[:, None]
                             + numpy.maximum(nu, 0)[None, :]})
        # the square roots of the M_b, for the perturbations of the data
        lam, Q = numpy.linalg.eigh(bM)
        self.Msqrt = Q * numpy.sqrt(numpy.maximum(lam, 0))[:, None, :]
        return


    def _split(self, x):
        k, nx = x.shape[0], 2 * self.N * self.J
        X = x[:, :nx].reshape(k, 2 * self.N, self.J)
        Xi = x[:, nx:].reshape(k, self.B, self.J) if self.slip.groups else None
        return X, Xi


    def _join(self, X, Xi):
        k = X.shape[0]
        flat = X.reshape(k, -1)
        return flat if Xi is None else numpy.concatenate([flat, Xi.reshape(k, -1)], axis=1)


    def apply(self, x):
        """
        A x for each row of {x}
        """
        s = self.slip
        X, Xi = self._split(x)
        T = numpy.matmul(s.G, X)
        if Xi is not None:
            T = T + Xi
        W = numpy.einsum("bij,kbj->kbi", self.bM, T)
        oX = numpy.matmul(s.G.T, W) + numpy.matmul(s.P, X) * s.dinv
        if Xi is None:
            return self._join(oX, None)
        oXi = W.copy()
        for g in s.groups:
            i = g["idx"]
            oXi[:, i] += numpy.matmul(numpy.matmul(g["Sinv"], Xi[:, i]), g["Pm"])
        return self._join(oX, oXi)


    def rhs(self):
        """
        P mu + beta G^T C^-1 d: the data term, the prior mean being zero
        """
        s = self.slip
        bh = self.beta * s.h
        return self._join((s.G.T @ bh)[None], bh[None] if s.groups else None)[0]


    def perturb(self, rows, rng):
        """
        {rows} draws of N(0, A): the data's, sqrt(beta) M_b^1/2 z per block, and the prior's,
        P^1/2 Z D^1/2 for X and S^-1/2 Z Pm^1/2 for Xi
        """
        s = self.slip
        N, J, B = self.N, self.J, self.B
        z = rng.standard_normal((rows, B, J))
        e = numpy.einsum("bij,kbj->kbi", self.Msqrt, z)
        eX = numpy.matmul(s.G.T, e)
        z2 = rng.standard_normal((rows, 2 * N, J))
        a = solve_triangular(s.LC, z2[:, :N].transpose(1, 0, 2).reshape(N, -1), lower=True,
                             trans="T")
        b = solve_triangular(s.LC, z2[:, N:].transpose(1, 0, 2).reshape(N, -1), lower=True,
                             trans="T") / s.rho
        prior = numpy.concatenate([a.reshape(N, rows, J), b.reshape(N, rows, J)]).transpose(1, 0, 2)
        eX = eX + prior * numpy.sqrt(s.dinv)
        if not s.groups:
            return self._join(eX, None)
        ei = e.copy()
        for g in s.groups:
            i = g["idx"]
            u = solve_triangular(g["LS"], rng.standard_normal((i.size, rows * J)), lower=True,
                                 trans="T")
            ei[:, i] += numpy.matmul(u.reshape(i.size, rows, J).transpose(1, 0, 2), g["Rm"].T)
        return self._join(eX, ei)


    def precondition(self, r):
        """
        The Kronecker approximation of A^-1 applied to each row of {r}
        """
        R, Ri = self._split(r)
        Z = self.U @ ((self.U.T @ R @ self.V) / self.denom) @ self.V.T
        if Ri is None:
            return self._join(Z, None)
        Zi = numpy.zeros_like(Ri)
        for g, p in zip(self.slip.groups, self.pre):
            i = g["idx"]
            Zi[:, i] = p["E"] @ ((p["E"].T @ Ri[:, i] @ p["V"]) / p["denom"]) @ p["V"].T
        return self._join(Z, Zi)


# end of file
