#!/usr/bin/env python3
# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# the slip history Mb of the kinematic model, with the arrival times from the triangle mesh: the
# arrival times at the source points of each patch, by barycentric interpolation, and the
# triangular source time functions of {castBigM_batched}; with its adjoint
import numpy
from mesh import Eikonal


class SlipHistory:
    """
    Mb[Nt][2][patches], flattened, for theta = [strike slips, dip slips, rise times, rupture
    velocities, hypocenter], each over {patches}, with the arrival times from {mesh}
    """

    def __init__(self, mesh, points, patch_of_point, patches, Nt, dt, t0s, Npt2, **eikonal):
        self.mesh, self.Np, self.Nt, self.dt = mesh, patches, Nt, dt
        self.t0s = numpy.asarray(t0s, dtype=float)
        self.Npt2 = Npt2
        self.eikonal = Eikonal(mesh, **eikonal)
        tri, self.weights = mesh.locate(points)
        self.corners = mesh.triangles[tri]
        self.point_patch = numpy.asarray(patch_of_point)
        return


    def __call__(self, θ):
        Np = self.Np
        T = self.eikonal.solve(θ[3 * Np:4 * Np], θ[4 * Np:4 * Np + 2])
        TI0 = (self.weights * T[self.corners]).sum(1)
        c, self._stf = self.stf(TI0=TI0, h=θ[2 * Np:3 * Np] / 2.0)
        self._θ = θ
        # (patches x Nt) -> Mb[Nt][2][patches]
        return numpy.stack([(c * θ[:Np, None]).T, (c * θ[Np:2 * Np, None]).T], axis=1).ravel()


    def stf(self, TI0, h):
        """
        c[patch, time]: the share of the patch's slip in each time interval, the triangles of the
        source points summed
        """
        p = self.point_patch
        hp = h[p][:, None]
        TI = TI0[:, None] + hp
        t = self.t0s[p][:, None] + self.dt * numpy.arange(self.Nt)[None, :]
        inside = ~((t <= TI - hp) | (t >= TI + hp))
        tri = numpy.where(inside, (1.0 - numpy.abs(TI - t) / hp) / hp / self.Npt2, 0.0)
        c = numpy.zeros((self.Np, self.Nt))
        numpy.add.at(c, p, tri)
        return c, (TI, t, hp, inside)


    def adjoint(self, gMb):
        """
        The gradient with respect to theta of a function of Mb, from its gradient {gMb}
        """
        Np, Nt = self.Np, self.Nt
        θ = self._θ
        g = gMb.reshape(Nt, 2, Np)
        TI, t, hp, inside = self._stf
        # Mb = c * slip
        cc = numpy.zeros((Np, Nt))
        p = self.point_patch
        tri = numpy.where(inside, (1.0 - numpy.abs(TI - t) / hp) / hp / self.Npt2, 0.0)
        numpy.add.at(cc, p, tri)
        gθ = numpy.zeros_like(θ)
        gθ[:Np] = (cc * g[:, 0, :].T).sum(1)
        gθ[Np:2 * Np] = (cc * g[:, 1, :].T).sum(1)
        gc = g[:, 0, :].T * θ[:Np, None] + g[:, 1, :].T * θ[Np:2 * Np, None]
        # the triangles: (1/h - |TI - t|/h^2) / Npt2, with TI = TI0 + h
        sign = numpy.sign(TI - t)
        gtri = numpy.where(inside, gc[p], 0.0) / self.Npt2
        gTI0 = (gtri * (-sign / hp ** 2)).sum(1)
        gh_point = (gtri * (-1.0 / hp ** 2 - sign / hp ** 2 + 2.0 * numpy.abs(TI - t) / hp ** 3)).sum(1)
        gh = numpy.zeros(Np)
        numpy.add.at(gh, p, gh_point)
        gθ[2 * Np:3 * Np] = gh / 2.0
        # the interpolation, then the eikonal
        gT = numpy.zeros(self.mesh.vertices.shape[0])
        numpy.add.at(gT, self.corners, self.weights * gTI0[:, None])
        g_v, g_h = self.eikonal.adjoint(gT)
        gθ[3 * Np:4 * Np] = g_v
        gθ[4 * Np:4 * Np + 2] = g_h
        return gθ


def rectangle_sources(Nas, Ndd, dsp, Npt):
    """
    The Npt x Npt source points of each patch of the rectangular fault, in the order of TI0 in
    the cuda kernels, strike-major, dip fastest, and the patch of each
    """
    q = (numpy.arange(Npt) + 0.5) * dsp / Npt - 0.5 * dsp
    us = (numpy.arange(Nas)[:, None] * dsp + q[None, :]).ravel()
    vs = (numpy.arange(Ndd)[:, None] * dsp + q[None, :]).ravel()
    U, V = numpy.meshgrid(us, vs, indexing="ij")
    I, J = numpy.meshgrid(numpy.arange(Nas * Npt) // Npt, numpy.arange(Ndd * Npt) // Npt, indexing="ij")
    return numpy.stack([U.ravel(), V.ravel()], 1), (I * Ndd + J).ravel()


# end of file
