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

        # my columns of theta
        idx_map = numpy.arange(self.parameters) if self.idx_map is None else numpy.asarray(self.idx_map)
        if idx_map.size != 4 * patches + 2:
            application.error.log(
                f"the kinematic mesh model needs {4 * patches + 2} parameters for {patches} "
                f"patches, got {idx_map.size}")
            raise SystemExit(1)
        self.gidx_map = altar.cuda.vector(source=idx_map.astype("int64"))
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


# end of file
