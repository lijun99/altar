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
# my base classes
from altar.models.BayesianL2 import BayesianL2
from .Kinematic import Kinematic


# declaration
class KinematicMesh(Kinematic, family="altar.models.seismic.kinematicmesh"):
    """
    The kinematic slip model on a triangulated fault: the rupture fronts from the hypocenter by
    the fast iterative method on a mesh refined from the patches, the slips of each patch over
    time, then data = Gb Mb

    cuda only. The patches and their source points come from {mesh}, as csi's
    TriangularPatchesKin saves them; my parameters, in {idx_map} order, are the strike slips, dip
    slips, rise times and rupture velocities of the patches, then the hypocenter along strike and
    down dip, in the surface coordinates of the mesh. The settings of the rectangular model, Nas,
    Ndd, Nmesh, dsp and Npt, don't apply.
    """


    # user configurable state
    mesh = altar.properties.path(default="mesh.h5")
    mesh.doc = "the patches: vertices, surface coordinates uv, faces, and the source points"

    refine = altar.properties.int(default=10)
    refine.doc = "the triangles of the arrival times along each edge of a patch"

    velocity_groups = altar.properties.str(default=None)
    velocity_groups.doc = "the dataset of {mesh} with the velocity group of each patch, which then " \
                          "share one rupture velocity; default a velocity per patch"


    # protocol obligations
    @altar.export
    def initialize(self, application):
        """
        Initialize the state of the model given a {problem} specification
        """
        # cuda only
        if altar.backends.active() != "cuda":
            application.error.log(
                "the kinematic mesh model has only a cuda implementation; run it with job.gpus >= 1")
            raise SystemExit(1)

        # mounts my inputs, loads the data and its covariance, lays out my psets
        BayesianL2.initialize(self, application=application)

        from altar.models.seismic.ext import cudaseismic
        from . import mesh as meshes
        import h5py

        # the patches and their source points
        with h5py.File(self.ifs[str(self.mesh)].uri.path, "r") as h5:
            vertices = numpy.asarray(h5["vertices"], dtype=float)
            uv = numpy.asarray(h5["uv"], dtype=float)
            faces = numpy.asarray(h5["faces"], dtype=int)
            sources = numpy.asarray(h5["sources/uv"], dtype=float)
            owners = numpy.asarray(h5["sources/patch"], dtype=int)
        patches = faces.shape[0]
        counts = numpy.bincount(owners, minlength=patches)
        if counts.min() != counts.max() or numpy.any(numpy.diff(owners) < 0):
            application.error.log(f"'{self.mesh}': each patch needs the same number of source "
                                  f"points, stored patch by patch")
            raise SystemExit(1)
        self.patches = patches
        self.NGbparameters = 2 * patches * self.Nt

        # the velocity group of each patch
        if self.velocity_groups is None:
            groups = numpy.arange(patches)
        else:
            with h5py.File(self.ifs[str(self.mesh)].uri.path, "r") as h5:
                groups = numpy.asarray(h5[self.velocity_groups], dtype=int)
        velocities = int(groups.max()) + 1
        if groups.size != patches or numpy.unique(groups).size != velocities:
            application.error.log(f"'{self.velocity_groups}': a group for each of the {patches} "
                                  f"patches, numbered from 0 without gaps")
            raise SystemExit(1)
        self.groups = groups

        # my columns of theta: the slips and rise times of the patches, the velocity of each group,
        # the hypocenter; the cuda model reads a velocity per patch, its group's
        expected = 3 * patches + velocities + 2
        columns = numpy.arange(self.parameters) if self.idx_map is None else numpy.asarray(self.idx_map)
        if columns.size != expected:
            application.error.log(
                f"the kinematic mesh model needs {expected} parameters for {patches} patches and "
                f"{velocities} rupture velocities, got {columns.size}")
            raise SystemExit(1)
        layout = numpy.concatenate([numpy.arange(3 * patches), 3 * patches + groups,
                                    3 * patches + velocities + numpy.arange(2)])
        self.gidx_map = altar.cuda.vector(source=columns[layout].astype("int64"))
        t0s = numpy.zeros(patches) if self.t0s is None else numpy.asarray(self.t0s, dtype=float)
        self.gt0s = altar.cuda.vector(source=t0s, dtype=self.precision)

        # the mesh of the arrival times, and its tables
        xyz, coords, triangles, patch = meshes.refine(vertices, faces, uv, self.refine)
        edges = xyz[triangles[:, [1, 2, 0]]] - xyz[triangles]
        spacing = float(numpy.sqrt((edges ** 2).sum(-1)).mean())
        corners, weights = meshes.locate(sources, owners, coords, triangles, patch)
        tables = meshes.tables(xyz, triangles, patch)
        self.vertices = xyz.shape[0]
        application.info.log(f"kinematic mesh: {patches} patches, {self.vertices} vertices,"
                             f" spacing {spacing:.3g} km, {counts[0]} source points per patch")

        # the green's functions, kept raw for the forward problem, and premerged with the data
        # covariance for sampling
        self.GF = self.io.load(filename=self.green, shape=(self.NGbparameters, self.observations))
        self.gGF = altar.cuda.matrix(source=self.GF, dtype=self.precision)
        self._premerge_covariance()

        self.cmodel = cudaseismic.cudaKinematicMesh(
            Np=patches, Nt=self.Nt, dt=self.dt, t0s=self.gt0s.grid, xyz=xyz,
            patchGeometry=meshes.patch_geometry(vertices, faces, uv), spacing=spacing,
            Nsp=int(counts[0]), corners=corners, weights=weights, samples=self.samples,
            parameters=self.parameters, observations=self.observations,
            idx_map=self.gidx_map.grid, **tables)
        # all done
        return self


    # private data
    patches = None
    vertices = None
    groups = None


# end of file
