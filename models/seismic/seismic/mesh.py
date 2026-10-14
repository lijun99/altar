# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# the triangulated fault of the kinematic model on a mesh: the patches, as csi saves them, refined
# for the arrival times, and the tables the cuda model reads

# externals
import numpy


def refine(vertices, faces, uv, m):
    """
    Each triangle of {faces} cut into m x m triangles, which keep its index as their patch; the
    shared vertices of neighboring triangles merged. Returns the vertices, their surface
    coordinates, the triangles and the patch of each
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
    points = numpy.concatenate([w @ vertices[f] for f in faces])
    coords = numpy.concatenate([w @ uv[f] for f in faces])
    triangles = numpy.concatenate([small + k * len(lattice) for k in range(len(faces))])
    # merge the copies of the vertices on shared edges
    key = numpy.round(points / (1e-6 * numpy.abs(points).max()))
    _, first, inverse = numpy.unique(key, axis=0, return_index=True, return_inverse=True)
    patch = numpy.repeat(numpy.arange(len(faces)), len(small))
    return points[first], coords[first], inverse.ravel()[triangles], patch


def csr(rows, values, count):
    """
    The pointers and the values of {values} grouped by {rows}, for {count} rows
    """
    order = numpy.argsort(rows, kind="stable")
    pointers = numpy.concatenate([[0], numpy.cumsum(numpy.bincount(rows, minlength=count))])
    return pointers.astype(numpy.int32), [v[order] for v in values]


def tables(vertices, triangles, patch):
    """
    For each vertex: the triangles it is a corner of, with their other two corners and patches;
    its neighbors; and the patches of its triangles, all as CSR
    """
    V = vertices.shape[0]
    t = triangles
    corner = numpy.concatenate([t[:, 0], t[:, 1], t[:, 2]])
    a = numpy.concatenate([t[:, 1], t[:, 2], t[:, 0]])
    b = numpy.concatenate([t[:, 2], t[:, 0], t[:, 1]])
    p = numpy.tile(patch, 3)
    incPtr, (incA, incB, incPatch) = csr(corner, [a, b, p], V)
    pairs = numpy.unique(numpy.concatenate([numpy.stack([corner, a], 1), numpy.stack([corner, b], 1)]), axis=0)
    nbPtr, (nb,) = csr(pairs[:, 0], [pairs[:, 1]], V)
    vp = numpy.unique(numpy.stack([corner, p], 1), axis=0)
    vpPtr, (vpPatch,) = csr(vp[:, 0], [vp[:, 1]], V)
    return {"incPtr": incPtr, "incA": incA, "incB": incB, "incPatch": incPatch,
            "nbPtr": nbPtr, "nb": nb, "vpPtr": vpPtr, "vpPatch": vpPatch}


def patch_geometry(vertices, faces, uv):
    """
    For each patch: the surface coordinates of its corners (6), the point of its first corner
    (3) and the 3 x 2 jacobian of its map from surface coordinates to the fault (6)
    """
    rows = []
    for f in faces:
        a, b, c = f
        M = numpy.column_stack([uv[b] - uv[a], uv[c] - uv[a]])
        E = numpy.column_stack([vertices[b] - vertices[a], vertices[c] - vertices[a]])
        J = E @ numpy.linalg.inv(M)
        rows.append(numpy.concatenate([uv[a], uv[b], uv[c], vertices[a], J.ravel()]))
    return numpy.asarray(rows)


def locate(points, patches, uv, triangles, patch):
    """
    The triangle of the refined mesh holding each source point, among those of its patch, and its
    barycentric weights
    """
    corners = numpy.zeros((len(points), 3), dtype=numpy.int32)
    weights = numpy.zeros((len(points), 3))
    for p in numpy.unique(patches):
        mine = numpy.flatnonzero(patch == p)
        A, B, C = (uv[triangles[mine, k]] for k in range(3))
        v0, v1 = B - A, C - A
        den = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]
        for q in numpy.flatnonzero(patches == p):
            d = points[q] - A
            l1 = (d[:, 0] * v1[:, 1] - d[:, 1] * v1[:, 0]) / den
            l2 = (v0[:, 0] * d[:, 1] - v0[:, 1] * d[:, 0]) / den
            l0 = 1 - l1 - l2
            k = numpy.minimum(numpy.minimum(l0, l1), l2).argmax()
            corners[q] = triangles[mine[k]]
            weights[q] = (l0[k], l1[k], l2[k])
    return corners, weights


# end of file
