#!/usr/bin/env python3
# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# the eikonal solver on two faults that meet: a splay ending on a main fault (T, three triangles
# on each edge of the junction) or crossing it (X, four); the arrival times at a constant velocity
# against the shortest paths on the two planes, which unfold into straight lines about the
# junction; the adjoint against central differences there; and the faults disconnected by a small
# offset, which the rupture can't cross. Run from this directory: python check_junction.py
import numpy
from mesh import Eikonal, junction_fault, subdivide

SIZE, MAIN, SPLAY, ANGLE, V = 5.0, (12, 4), 6, 30.0, 3.0
yj = (MAIN[0] // 2) * SIZE
rng = numpy.random.default_rng(3)


def build(kind, refine, offset=0.0):
    vertices, uv, faces, fault = junction_fault(kind, SIZE, MAIN, SPLAY, ANGLE, offset)
    return subdivide(vertices, faces, numpy.arange(len(faces)), uv, refine), fault


def edges(mesh):
    """
    How many edges have 1, 2, 3, ... triangles
    """
    t = mesh.triangles
    e = numpy.sort(numpy.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1)
    _, counts = numpy.unique(e, axis=0, return_counts=True)
    return {int(k): int(n) for k, n in zip(*numpy.unique(counts, return_counts=True))}


def exact(mesh, H, on_splay):
    """
    The shortest path from {H} to each vertex over the two planes: straight within the fault of
    the hypocenter, unfolded about the junction to the other
    """
    X = mesh.vertices
    main = numpy.abs(X[:, 0]) < 1e-9
    # the distance to the junction line within each fault, and along it
    to_line = numpy.hypot(X[:, 0], X[:, 1] - yj)
    h_line = numpy.hypot(H[0], H[1] - yj)
    straight = numpy.sqrt(((X - H) ** 2).sum(1))
    unfolded = numpy.hypot(h_line + to_line, X[:, 2] - H[2])
    same = ~main if on_splay else main
    return numpy.where(same, straight, unfolded), main


print(f"two vertical faults, {SIZE:.0f} km patches: main {MAIN[0] * SIZE:.0f} x {MAIN[1] * SIZE:.0f} km,"
      f" splay {SPLAY * SIZE:.0f} km at {ANGLE:.0f} degrees, velocity {V} km/s")
print("1. arrival times against the shortest paths, by fault and side of the junction")
for kind in ["T", "X"]:
    for refine in [5, 10, 20]:
        mesh, fault = build(kind, refine)
        Np = int(mesh.patch.max()) + 1
        eik = Eikonal(mesh)
        # the hypocenter on the splay, 20 km from the junction; then on the main fault, 15 km from it
        for label, h in [("from the splay", (220.0, 10.0)), ("from the main fault", (15.0, 10.0))]:
            T = eik.solve(numpy.full(Np, V), numpy.array(h))
            H = mesh.point(numpy.array(h))[0]
            D, main = exact(mesh, H, on_splay=h[0] > 100)
            err = numpy.abs(T - D / V)
            far = D > 5.0
            parts = {"splay": ~main, "main, y < junction": main & (mesh.vertices[:, 1] < yj - 1e-9),
                     "main, y > junction": main & (mesh.vertices[:, 1] > yj + 1e-9)}
            cells = "; ".join(f"{name} max {err[m].max():.3f} s, rel {numpy.median(err[m & far] / T[m & far]):.1e}"
                              for name, m in parts.items())
            print(f"   {kind} refine {refine:2d} {label:20s} spacing {mesh.spacing:.2f} km: {cells}")
        if refine == 5:
            print(f"   {kind}: edges by number of triangles {edges(mesh)}")

print("2. the adjoint against central differences, random velocities, a hypocenter on each fault")
for kind in ["T", "X"]:
    mesh, fault = build(kind, 4)
    Np = int(mesh.patch.max()) + 1
    eik = Eikonal(mesh)
    for h in [numpy.array([218.0, 9.0]), numpy.array([12.0, 8.0])]:
        v = rng.uniform(2.0, 3.5, Np)
        r = rng.standard_normal(mesh.vertices.shape[0])
        eik.solve(v, h)
        gv, gh = eik.adjoint(r)
        worst = 0.0
        for trial in range(3):
            dv, dh = rng.standard_normal(Np), rng.standard_normal(2)
            ε = 1e-6
            f = lambda e: r @ eik.solve(v + e * dv, h + e * dh)
            fd = (f(ε) - f(-ε)) / (2 * ε)
            an = gv @ dv + gh @ dh
            worst = max(worst, abs(an - fd) / abs(fd))
        eik.solve(v, h)
        eik.adjoint(r)
        print(f"   {kind}, hypocenter {h}: worst relative difference {worst:.1e},"
              f" causality violations {eik.violations}")

print("3. the splay moved off the junction by a small offset: not connected")
for offset in [0.0, 0.05]:
    mesh, fault = build("T", 5, offset)
    Np = int(mesh.patch.max()) + 1
    T = Eikonal(mesh).solve(numpy.full(Np, V), numpy.array([220.0, 10.0]))
    main = numpy.abs(mesh.vertices[:, 0]) < 1e-9
    print(f"   offset {offset * 1000:4.0f} m: {mesh.vertices.shape[0]} vertices, edges {edges(mesh)};"
          f" main fault arrivals {T[main].min():.1f}-{T[main].max():.1f} s")


# end of file
