#!/usr/bin/env python3
# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# rupture arrival times on a triangulated fault: the eikonal equation solved by the fast iterative
# method (Jeong & Whitaker 2008) on the mesh vertices, and its adjoint, a backward pass over the
# upwind parents of the converged solution
import numpy


class Mesh:
    """
    A triangulated fault surface: {vertices} (V x 3), {triangles} (T x 3), the parameter patch of
    each triangle, and the surface coordinates (along strike, down dip) of each vertex (V x 2), or
    of the corners of each triangle (T x 3 x 2), where faults meet and a vertex has one per fault
    """

    def __init__(self, vertices, triangles, patch, uv):
        self.vertices = numpy.asarray(vertices, dtype=float)
        self.triangles = numpy.asarray(triangles, dtype=int)
        self.patch = numpy.asarray(patch, dtype=int)
        uv = numpy.asarray(uv, dtype=float)
        per_vertex = uv.ndim == 2
        self.uv = uv if per_vertex else None
        self.corner_uv = uv[self.triangles] if per_vertex else uv
        V = self.vertices.shape[0]
        # the corners: each vertex of each triangle, with the other two, as the update reads them
        t = self.triangles
        self.corner = numpy.concatenate([t[:, 0], t[:, 1], t[:, 2]])
        self.other = numpy.concatenate([t[:, [1, 2]], t[:, [2, 0]], t[:, [0, 1]]])
        self.corner_triangle = numpy.tile(numpy.arange(t.shape[0]), 3)
        # the vertices sharing a triangle with each vertex
        pairs = numpy.concatenate([numpy.stack([self.corner, self.other[:, 0]], 1),
                                   numpy.stack([self.corner, self.other[:, 1]], 1)])
        pairs = numpy.unique(pairs, axis=0)
        self.neighbor_ptr = numpy.searchsorted(pairs[:, 0], numpy.arange(V + 1))
        self.neighbor = pairs[:, 1]
        # the edge lengths of each triangle, for the spacing
        e = self.vertices[t[:, [1, 2, 0]]] - self.vertices[t]
        self.spacing = float(numpy.sqrt((e ** 2).sum(-1)).mean())
        return


    def neighbors(self, mask):
        """
        The vertices next to those in {mask}
        """
        out = numpy.zeros(mask.size, dtype=bool)
        rows = numpy.flatnonzero(mask)
        if rows.size:
            idx = numpy.concatenate([self.neighbor[self.neighbor_ptr[r]:self.neighbor_ptr[r + 1]]
                                     for r in rows])
            out[idx] = True
        return out


    def locate(self, points):
        """
        The triangle containing each of the (P x 2) surface {points}, and its barycentric weights
        """
        P = numpy.atleast_2d(points)
        a, b, c = (self.corner_uv[:, k] for k in range(3))
        v0, v1 = b - a, c - a
        d = P[:, None, :] - a[None, :, :]
        den = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]
        l1 = (d[..., 0] * v1[:, 1] - d[..., 1] * v1[:, 0]) / den
        l2 = (v0[:, 0] * d[..., 1] - v0[:, 1] * d[..., 0]) / den
        l0 = 1 - l1 - l2
        worst = numpy.minimum(numpy.minimum(l0, l1), l2)
        tri = worst.argmax(axis=1)
        rows = numpy.arange(P.shape[0])
        return tri, numpy.stack([l0[rows, tri], l1[rows, tri], l2[rows, tri]], 1)


    def point(self, uv):
        """
        The point of the surface at the surface coordinates {uv}, and its (3 x 2) derivative,
        from the affine map of the triangle that holds it
        """
        tri, w = self.locate(numpy.asarray(uv, dtype=float)[None, :])
        a, b, c = self.triangles[tri[0]]
        X = w[0] @ self.vertices[[a, b, c]]
        ua, ub, uc = self.corner_uv[tri[0]]
        M = numpy.column_stack([ub - ua, uc - ua])
        E = numpy.column_stack([self.vertices[b] - self.vertices[a], self.vertices[c] - self.vertices[a]])
        return X, E @ numpy.linalg.inv(M)


def subdivide(vertices, triangles, patch, uv, m):
    """
    The mesh of {triangles}, each cut into m x m triangles, which keep its patch; the vertices at
    the same point merged, so faults that meet share them, each triangle keeping its own surface
    coordinates
    """
    vertices, uv = numpy.asarray(vertices, dtype=float), numpy.asarray(uv, dtype=float)
    lattice = [(i, j) for i in range(m + 1) for j in range(m + 1 - i)]
    index = {ij: k for k, ij in enumerate(lattice)}
    L = numpy.array(lattice, dtype=float) / m
    w = numpy.column_stack([1 - L.sum(1), L])
    small = []
    for i in range(m):
        for j in range(m - i):
            small.append((index[i, j], index[i + 1, j], index[i, j + 1]))
            if i + j < m - 1:
                small.append((index[i + 1, j], index[i + 1, j + 1], index[i, j + 1]))
    small = numpy.array(small)
    points = numpy.concatenate([w @ vertices[t] for t in triangles])
    coords = numpy.concatenate([w @ uv[t] for t in triangles])
    faces = numpy.concatenate([small + k * len(lattice) for k in range(len(triangles))])
    # merge the copies of the vertices on shared edges
    key = numpy.round(points / (1e-6 * numpy.abs(points).max()))
    _, first, inverse = numpy.unique(key, axis=0, return_index=True, return_inverse=True)
    return Mesh(vertices=points[first], triangles=inverse.ravel()[faces],
                patch=numpy.repeat(numpy.asarray(patch), len(small)), uv=coords[faces])


def rectangle_fault(Nas, Ndd, dsp, n, shape=None):
    """
    The fault of {Nas} x {Ndd} square patches of side {dsp}, each cut into {n} x {n} squares and
    each square into two triangles, the diagonals alternating; the surface coordinates put the
    origin at the center of the first patch, as the hypocenter has it; planar, or bent by
    {shape}, a map from the surface coordinates to the points of the fault
    """
    u = -0.5 * dsp + numpy.arange(Nas * n + 1) * dsp / n
    v = -0.5 * dsp + numpy.arange(Ndd * n + 1) * dsp / n
    U, W = numpy.meshgrid(u, v, indexing="ij")
    uv = numpy.stack([U.ravel(), W.ravel()], 1)
    vertices = numpy.column_stack([uv, numpy.zeros(len(uv))]) if shape is None else shape(uv)
    index = lambda i, j: i * (Ndd * n + 1) + j
    triangles, patch = [], []
    for i in range(Nas * n):
        for j in range(Ndd * n):
            a, b, c, d = index(i, j), index(i + 1, j), index(i + 1, j + 1), index(i, j + 1)
            if (i + j) % 2:
                triangles += [(a, b, c), (a, c, d)]
            else:
                triangles += [(a, b, d), (b, c, d)]
            patch += [(i // n) * Ndd + j // n] * 2
    return Mesh(vertices=vertices, triangles=triangles, patch=patch, uv=uv)


def junction_fault(kind="T", size=5.0, main=(6, 3), splay=3, angle=30.0, offset=0.0):
    """
    Two vertical faults that meet along a vertical line, as csi saves them: a main fault in the
    plane x = 0, {main} squares of {size} along strike (y) and down dip, and a splay at {angle}
    to it from the line at the middle of the main fault, {splay} squares long, ending on it ("T")
    or crossing it ("X"); each square cut into two triangles, the diagonals alternating. Each fault
    keeps its own copy of the vertices on the line, with its own surface coordinates, the splay's
    from 200 km along strike so the two don't overlap; {offset} (km) moves the splay off the line,
    disconnecting the faults. Returns the vertices, uv, faces and the fault of each face (0, 1)
    """
    ns, nd = main
    yj = (ns // 2) * size
    α = numpy.radians(angle)
    lo = -splay if kind == "X" else 0
    sheets = [
        # the main fault: along strike y, down dip z
        (lambda i, j: ((0.0, i * size, 1.0 + j * size), (i * size, j * size)), ns),
        # the splay: s from the line, at {angle} to the main fault's strike
        (lambda i, j: ((offset + (lo + i) * size * numpy.sin(α), yj - (lo + i) * size * numpy.cos(α),
                        1.0 + j * size), (200.0 + (lo + i) * size, j * size)), splay - lo),
    ]
    vertices, uv, faces, fault = [], [], [], []
    for k, (at, n) in enumerate(sheets):
        base = len(vertices)
        for i in range(n + 1):
            for j in range(nd + 1):
                x, w = at(i, j)
                vertices.append(x)
                uv.append(w)
        index = lambda i, j: base + i * (nd + 1) + j
        for i in range(n):
            for j in range(nd):
                a, b, c, d = index(i, j), index(i + 1, j), index(i + 1, j + 1), index(i, j + 1)
                faces += [(a, b, c), (a, c, d)] if (i + j) % 2 else [(a, b, d), (b, c, d)]
                fault += [k, k]
    return numpy.array(vertices), numpy.array(uv), numpy.array(faces), numpy.array(fault)


def source_points(uv, faces, n):
    """
    The centroids of the n x n sub-triangles of each face, in surface coordinates, and their face
    """
    bary = []
    for i in range(n):
        for j in range(n - i):
            bary.append(((i + 1 / 3) / n, (j + 1 / 3) / n))
            if i + j < n - 1:
                bary.append(((i + 2 / 3) / n, (j + 2 / 3) / n))
    w = numpy.column_stack([1 - numpy.sum(bary, 1), bary])
    return numpy.concatenate([w @ uv[f] for f in faces]), numpy.repeat(numpy.arange(len(faces)), len(w))


def update(TA, TB, A, B, C, s):
    """
    The upwind update of C from the edge AB of a triangle with slowness {s}: the time, and its
    derivatives with respect to TA, TB and s; by the envelope theorem, the interpolation weights
    of the point P the ray crosses AB at, and the length of PC
    """
    w, e = C - A, B - A
    a = (e * e).sum(-1)
    b = (w * e).sum(-1)
    c = (w * w).sum(-1)
    D = numpy.maximum(a * c - b * b, 0.0)
    # the one-point updates, from A and from B
    lA, lB = numpy.sqrt(c), numpy.sqrt(((C - B) ** 2).sum(-1))
    tA, tB = TA + s * lA, TB + s * lB
    fromA = tA <= tB
    T = numpy.where(fromA, tA, tB)
    wA, wB = fromA.astype(float), (~fromA).astype(float)
    ws = numpy.where(fromA, lA, lB)
    # the two-point update, where the ray crosses the inside of AB
    k = (TB - TA) / s
    ok = (k * k < a) & numpy.isfinite(k)
    kk = numpy.where(ok, k, 0.0)
    μ = numpy.sign(kk) * numpy.abs(kk) * numpy.sqrt(D / numpy.where(ok, a - kk * kk, 1.0))
    λ = (b - μ) / a
    length = numpy.sqrt((μ * μ + D) / a)
    t2 = TA + λ * (TB - TA) + s * length
    inside = ok & (λ > 0) & (λ < 1) & (t2 < T)
    T = numpy.where(inside, t2, T)
    wA = numpy.where(inside, 1 - λ, wA)
    wB = numpy.where(inside, λ, wB)
    ws = numpy.where(inside, length, ws)
    return T, wA, wB, ws


class Eikonal:
    """
    The arrival times of the rupture at the vertices of {mesh}, from a hypocenter and the rupture
    velocity of each patch, with the derivatives the adjoint needs
    """

    def __init__(self, mesh, radius=1.0, steepness=10.0, cap=1.0e6, tolerance=0.0):
        self.mesh = mesh
        self.radius, self.steepness, self.cap = radius, steepness, cap
        self.tolerance = tolerance
        return


    def seed(self, hypocenter, s_vertex):
        """
        The straight ray times from the hypocenter within {radius} mesh spacings, rising steeply
        past it, capped, as the structured solver seeds them; and their derivatives
        """
        mesh = self.mesh
        h = mesh.spacing
        # the hypocenter on the surface: the point of its triangle, and dX/d(surface coordinates)
        X, dX = mesh.point(hypocenter)
        r = mesh.vertices - X
        distance = numpy.sqrt((r * r).sum(1))
        far = numpy.maximum(0.0, distance - self.radius * h)
        path = distance + self.steepness * far * far / h
        T0 = path * s_vertex
        capped = T0 >= self.cap
        T0 = numpy.where(capped, self.cap, T0)
        # d T0 / d hypocenter, (V x 2), and d T0 / d s_vertex
        dd = -(r / numpy.maximum(distance, 1e-300)[:, None]) @ dX
        dpath = 1.0 + 2.0 * self.steepness * far / h
        dT0_dh = numpy.where(capped[:, None], 0.0, (s_vertex * dpath)[:, None] * dd)
        dT0_ds = numpy.where(capped, 0.0, path)
        return T0, dT0_dh, dT0_ds


    def solve(self, velocity, hypocenter):
        """
        The arrival times at the vertices, for the rupture {velocity} of each patch and the
        {hypocenter} in surface coordinates; keeps what the adjoint needs
        """
        mesh = self.mesh
        V = mesh.vertices.shape[0]
        s_tri = 1.0 / velocity[mesh.patch]
        # a vertex seeds with the fastest of its triangles
        s_vertex = numpy.full(V, numpy.inf)
        numpy.minimum.at(s_vertex, mesh.corner, s_tri[mesh.corner_triangle])
        T0, dT0_dh, dT0_ds = self.seed(hypocenter=hypocenter, s_vertex=s_vertex)
        # the fast iterative method, the active vertices updated together
        T = T0.copy()
        active = numpy.ones(V, dtype=bool)
        sweeps = 0
        while active.any():
            sweeps += 1
            rows = numpy.flatnonzero(active[mesh.corner])
            v, (a, b), t = mesh.corner[rows], mesh.other[rows].T, mesh.corner_triangle[rows]
            X = mesh.vertices
            new = update(T[a], T[b], X[a], X[b], X[v], s_tri[t])[0]
            best = numpy.full(V, numpy.inf)
            numpy.minimum.at(best, v, new)
            changed = best < T - self.tolerance
            T = numpy.where(changed, best, T)
            active = mesh.neighbors(changed)
        self.sweeps = sweeps
        # the parents: the update each vertex took at the converged solution
        X = mesh.vertices
        v, (a, b), t = mesh.corner, mesh.other.T, mesh.corner_triangle
        new, wA, wB, ws = update(T[a], T[b], X[a], X[b], X[v], s_tri[t])
        order = numpy.lexsort((new, v))
        first = order[numpy.r_[True, v[order][1:] != v[order][:-1]]]
        winner = numpy.full(V, -1)
        winner[v[first]] = first
        seeded = T0 <= new[winner]
        self.T, self.T0, self.seeded = T, T0, seeded
        self.parent = numpy.stack([a[winner], b[winner]], 1)
        self.weight = numpy.stack([wA[winner], wB[winner]], 1)
        self.length, self.parent_triangle = ws[winner], t[winner]
        self.s_tri, self.s_vertex = s_tri, s_vertex
        self.dT0_dh, self.dT0_ds = dT0_dh, dT0_ds
        self.velocity = numpy.asarray(velocity, dtype=float)
        return T


    def adjoint(self, gT):
        """
        From the gradient {gT} of a function of the arrival times, its gradient with respect to the
        rupture velocity of each patch and the hypocenter
        """
        mesh = self.mesh
        λ = numpy.array(gT, dtype=float)
        g_s_tri = numpy.zeros(mesh.triangles.shape[0])
        g_T0 = numpy.zeros_like(λ)
        T = self.T
        violations = 0
        # latest first: every vertex hands its adjoint to its parents before they are visited
        for v in numpy.argsort(-T, kind="stable"):
            if self.seeded[v]:
                g_T0[v] += λ[v]
                continue
            a, b = self.parent[v]
            wa, wb = self.weight[v]
            violations += (wa > 0 and T[a] >= T[v]) + (wb > 0 and T[b] >= T[v])
            λ[a] += wa * λ[v]
            λ[b] += wb * λ[v]
            g_s_tri[self.parent_triangle[v]] += self.length[v] * λ[v]
        self.violations = violations
        # the seeds: through the hypocenter, and the slowness each seed took from its triangles
        g_h = g_T0 @ self.dT0_dh
        g_s_vertex = g_T0 * self.dT0_ds
        # the slowness of a vertex is the least of its triangles'; credit the one that has it
        s_tri = self.s_tri
        owner = numpy.full(mesh.vertices.shape[0], -1)
        hit = s_tri[mesh.corner_triangle] == self.s_vertex[mesh.corner]
        owner[mesh.corner[hit]] = mesh.corner_triangle[hit]
        numpy.add.at(g_s_tri, owner, g_s_vertex)
        # slowness to velocity, triangle to patch
        g_velocity = numpy.zeros_like(self.velocity)
        numpy.add.at(g_velocity, mesh.patch, -s_tri ** 2 * g_s_tri)
        return g_velocity, g_h


# end of file
