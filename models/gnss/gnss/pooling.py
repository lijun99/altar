# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# the geometry of the spatially pooled sparsity prior ({altar.models.gnss.pooled}) on one group
# of a trajectory case: the stations of its blocks and their components, the pools of components
# that share a sparsity pattern, and the neighborhood weights W between stations
#
# for each column j of the group, each station s and each pool, the pooled energy is
#     e = sum_k W_sk sum_{c in pool} theta_kcj^2
# with W_sk a gaussian kernel of the distance between stations s and k, cut off at {radius};
# its columns are scaled so that sum_s sqrt(W_sk) = 1, which makes a coefficient that is active
# alone, with no active neighbor, cost what it would under a plain Laplace prior (with the
# "laplace" shape; with "log", for small coefficients)


# externals
import numpy

EARTH_KM = 6371.0
POOLINGS = {
    # each component on its own
    "components": None,
    # east and north together, up alone
    "horizontal": [("east", "north"), ("up",)],
    # all the components of a station together
    "all": "all",
}


def distances(lon, lat):
    """
    The great-circle distances (km) between all pairs of points
    """
    lo, la = numpy.radians(lon), numpy.radians(lat)
    dlo = lo[:, None] - lo[None, :]
    c = (numpy.sin(la)[:, None] * numpy.sin(la)[None, :]
         + numpy.cos(la)[:, None] * numpy.cos(la)[None, :] * numpy.cos(dlo))
    return EARTH_KM * numpy.arccos(numpy.clip(c, -1.0, 1.0))


class Geometry:
    """
    Stations, components, pools and neighborhood weights of a group of {width} columns
    """

    def __init__(self, station, component, lon, lat, width, length, radius, pooling):
        # the stations, in order of first appearance, and where each block belongs
        names = list(dict.fromkeys(station))
        comps = list(dict.fromkeys(component))
        self.stations, self.components = names, comps
        self.blocks = len(station)
        self.width = int(width)
        where = {name: i for i, name in enumerate(names)}
        which = {name: i for i, name in enumerate(comps)}
        # block[k, c]: the block of station k and component c, -1 if missing
        self.block = numpy.full((len(names), len(comps)), -1, dtype=numpy.int32)
        for b, (s, c) in enumerate(zip(station, component)):
            self.block[where[s], which[c]] = b
        # the pools of components
        if pooling not in POOLINGS:
            raise ValueError(f"pooled: pooling '{pooling}', not one of {', '.join(POOLINGS)}")
        spec = POOLINGS[pooling]
        if spec is None:
            pools = [[c] for c in comps]
        elif spec == "all":
            pools = [comps]
        else:
            pools = [[c for c in p if c in which] for p in spec]
            missing = [c for c in comps if not any(c in p for p in pools)]
            pools = [p for p in pools if p] + [[c] for c in missing]
        self.pools = [[which[c] for c in p] for p in pools]
        self.pool_of = numpy.zeros(len(comps), dtype=numpy.int32)
        for i, p in enumerate(self.pools):
            self.pool_of[p] = i
        # the station coordinates, from the first block of each
        first = [station.index(s) if isinstance(station, list) else list(station).index(s)
                 for s in names]
        self.lon = numpy.asarray(lon, dtype=float)[first]
        self.lat = numpy.asarray(lat, dtype=float)[first]
        # the neighborhood weights
        d = distances(self.lon, self.lat)
        W = numpy.where(d <= radius, numpy.exp(-0.5 * (d / length) ** 2), 0.0)
        W /= numpy.sqrt(W).sum(axis=0, keepdims=True) ** 2
        self.W = W
        # as compressed rows, of W and of its transpose, for the gpu
        self.rows = csr(W)
        self.cols = csr(W.T)
        return

    @property
    def neighbors(self):
        """
        The number of stations within the radius of each, itself included
        """
        return numpy.diff(self.rows[0])

    def describe(self):
        n = self.neighbors
        names = [self.components[p[0]] if len(p) == 1 else "+".join(self.components[c] for c in p)
                 for p in self.pools]
        return (f"{len(self.stations)} stations, {self.blocks} blocks, {self.width} columns; "
                f"pools {names}; neighbors per station median {int(numpy.median(n))}, "
                f"max {n.max()}")


def csr(M):
    """
    The compressed rows of the dense matrix {M}: row pointers, column indices, values
    """
    rows, cols = numpy.nonzero(M)
    indptr = numpy.zeros(M.shape[0] + 1, dtype=numpy.int32)
    numpy.add.at(indptr, rows + 1, 1)
    return (numpy.cumsum(indptr).astype(numpy.int32), cols.astype(numpy.int32),
            M[rows, cols].astype(float))


def load(path, parameters, length, radius, pooling):
    """
    The geometry of a group of {parameters} parameters of the case file {path}
    """
    import h5py
    with h5py.File(path, "r") as h5:
        text = lambda values: [v.decode() if isinstance(v, bytes) else str(v) for v in values]
        station = text(h5["blocks/station"][:])
        component = text(h5["blocks/component"][:])
        if "blocks/lon" not in h5:
            raise ValueError(f"pooled: '{path}' has no station coordinates, blocks/lon and lat")
        lon, lat = h5["blocks/lon"][:], h5["blocks/lat"][:]
    blocks = len(station)
    if parameters % blocks:
        raise ValueError(f"pooled: {parameters} parameters aren't a whole number of columns "
                         f"for the {blocks} blocks of '{path}'")
    return Geometry(station, component, lon, lat, parameters // blocks, length, radius, pooling)


# end of file
