# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# the slip histories of the kinematic model on a triangulated fault outside of an application, for
# programs that make forward models, e.g. csi

# externals
import numpy
# the package
from . import mesh as meshes


class SlipHistory:
    """
    The slips of each patch over time, Mb, and the rupture times, of the kinematic model on a
    triangulated fault, on the cpu: the patches {faces} of {vertices} (km), the surface
    coordinates {uv} of the vertices, the source points of each patch, {sources} in surface
    coordinates and their {patches}, stored patch by patch; {Nt} time steps of {dt} seconds; the
    rupture times on the patches cut {refine} x {refine} times; the source time function {stf},
    a triangle peaking at {peak} of the rise time, or a raised cosine

    The data are then big G @ Mb, with the big G of the model, one column per time step, rake
    and patch, the patch fastest
    """


    def __init__(self, vertices, faces, uv, sources, patches, Nt, dt, refine=10, stf="triangle",
                 peak=0.5):
        from .ext import seismic
        kinds = {"triangle": 0, "cosine": 1}
        if stf not in kinds or not 0 < peak < 1:
            raise ValueError(f"stf: triangle or cosine, and a peak in (0, 1); got '{stf}', {peak}")
        faces = numpy.asarray(faces, dtype=int)
        patches = numpy.asarray(patches, dtype=int)
        Np = faces.shape[0]
        counts = numpy.bincount(patches, minlength=Np)
        if counts.min() != counts.max() or numpy.any(numpy.diff(patches) < 0):
            raise ValueError("each patch needs the same number of source points, stored patch by patch")
        self.patches, self.Nt = Np, int(Nt)
        grid = meshes.arrival_mesh(numpy.asarray(vertices, dtype=float), faces,
                                   numpy.asarray(uv, dtype=float),
                                   numpy.asarray(sources, dtype=float), patches, refine)
        self.corners, self.weights = grid["corners"], grid["weights"]
        self.model = seismic.KinematicMesh(
            Np=Np, Nt=self.Nt, dt=float(dt), t0s=numpy.zeros(Np), Nsp=int(counts[0]),
            idx_map=numpy.arange(4 * Np + 2, dtype="int64"), precision="float64", **grid)
        self.model.set_stf(kind=kinds[stf], peak=float(peak))
        return


    def __call__(self, slip, risetime, velocity, hypocenter):
        """
        Mb (Nt x 2 x patches) from the {slip} of each patch along the two rakes of big G
        (patches x 2, m), the {risetime} (s) and rupture {velocity} (km/s) of each patch and the
        {hypocenter} in surface coordinates (km); for a batch, a leading axis on each of them and
        on Mb
        """
        theta, single = self._theta(slip, risetime, velocity, hypocenter)
        Mb = numpy.zeros((theta.shape[0], 2 * self.patches * self.Nt))
        self.model.cast_mb(theta=theta, mb=Mb, batch=theta.shape[0])
        Mb = Mb.reshape(-1, self.Nt, 2, self.patches)
        return Mb[0] if single else Mb


    def arrivals(self, velocity, hypocenter):
        """
        The rupture times (s) at the source points, and at the vertices of the refined mesh, from
        the rupture {velocity} of each patch (km/s) and the {hypocenter} in surface coordinates
        (km); for a batch, a leading axis on each of them and on the times
        """
        single = numpy.ndim(velocity) == 1 and numpy.ndim(hypocenter) == 1
        velocity = numpy.atleast_2d(velocity)
        theta, _ = self._theta(numpy.zeros(velocity.shape + (2,)), numpy.ones(velocity.shape),
                               velocity, hypocenter)
        T = numpy.asarray(self.model.arrivals(theta=theta, batch=theta.shape[0]))
        T = T.reshape(theta.shape[0], -1)
        sources = (self.weights * T[:, self.corners]).sum(-1)
        return (sources[0], T[0]) if single else (sources, T)


    def _theta(self, slip, risetime, velocity, hypocenter):
        """
        The rows of the compiled model: the slips along the two rakes, the rise times, the
        velocities and the hypocenter
        """
        slip = numpy.asarray(slip, dtype=float)
        single = slip.ndim == 2
        slip = slip.reshape(-1, self.patches, 2)
        parts = [slip[..., 0], slip[..., 1]] + [numpy.atleast_2d(numpy.asarray(x, dtype=float))
                                                for x in (risetime, velocity, hypocenter)]
        rows = max(p.shape[0] for p in parts)
        theta = numpy.concatenate([numpy.broadcast_to(p, (rows, p.shape[1])) for p in parts], axis=1)
        return numpy.ascontiguousarray(theta), single and rows == 1


# end of file
