# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#


# externals
import math
import numpy
import cuda.tile as ct
# the package
import altar
import altar.cuda
from altar.cuda import tile


# the tile shape: samples x observations
TS = 8
TO = 64
# the entries of the kernel {constants}
PI, DEG, EPS, NU = range(4)
# the side table: for each of the twelve sides of the three rectangles, its end points, the
# burgers vector of its rectangle, and whether the rectangle contributes
SIDES = 12
PAX, PAY, PAZ, PBX, PBY, PBZ, BX, BY, BZ, ACTIVE, FIELDS = range(11)


# the compound dislocation model of Nikkhoo et al. [2017], as in {lib/libcdm/cdm.h}; source
# quantities are (TS, 1) tiles, quantities at the observation points are (TS, TO) tiles, and
# vectors are tuples of three tiles; always in double precision, since the angular dislocations
# cancel too much for single precision (errors up to ~20% of the signal), whatever the
# precision {theta} and {predicted} are stored in

def add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

def scale(s, a):
    return (s * a[0], s * a[1], s * a[2])

def cross(a, b):
    return (a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0])

def norm(a):
    return ct.sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2])

def select(condition, a, b):
    return (ct.where(condition, a[0], b[0]), ct.where(condition, a[1], b[1]),
            ct.where(condition, a[2], b[2]))


def source(p, constants):
    """
    The twelve vertices of the three rectangular dislocations, and the axes, of the sources with
    parameters {p}, laid out as in {parameter_t}
    """
    deg = tile.constant(constants, DEG)
    sx = ct.sin(p[7]*deg)
    cx = ct.cos(p[7]*deg)
    sy = ct.sin(p[8]*deg)
    cy = ct.cos(p[8]*deg)
    sz = ct.sin(p[9]*deg)
    cz = ct.cos(p[9]*deg)
    # the columns of R = Rz Ry Rx
    R0 = (cz*cy, -sz*cy, sy)
    R1 = (cz*sy*sx + sz*cx, -sz*sy*sx + cz*cx, -cy*sx)
    R2 = (-cz*sy*cx + sz*sx, sz*sy*cx + cz*sx, cy*cx)
    # the axes
    ax = 2*p[4]
    ay = 2*p[5]
    az = 2*p[6]
    # the centroid
    P0 = (p[0], p[1], -p[2])

    P1 = add(P0, add(scale(0.5*ay, R1), scale(0.5*az, R2)))
    P2 = sub(P1, scale(ay, R1))
    P3 = sub(P2, scale(az, R2))
    P4 = sub(P1, scale(az, R2))

    Q1 = add(P0, sub(scale(0.5*az, R2), scale(0.5*ax, R0)))
    Q2 = add(Q1, scale(ax, R0))
    Q3 = sub(Q2, scale(az, R2))
    Q4 = sub(Q1, scale(az, R2))

    S1 = add(P0, add(scale(0.5*ax, R0), scale(0.5*ay, R1)))
    S2 = sub(S1, scale(ax, R0))
    S3 = sub(S2, scale(ay, R1))
    S4 = sub(S1, scale(ay, R1))

    return (P1, P2, P3, P4), (Q1, Q2, Q3, Q4), (S1, S2, S3, S4), (ax, ay, az)


def buried(planes):
    """
    Whether each source lies below the free surface
    """
    ok = planes[0][0][2] <= 0
    for plane in ct.static_iter(planes):
        for vertex in ct.static_iter(plane):
            ok = ok & (vertex[2] <= 0)
    return ok


def AngDisDispSurf(y1, y2, beta, b, nu, a, constants):
    """
    The surface displacements of an angular dislocation, in its own coordinate system
    """
    pi = tile.constant(constants, PI)
    sinB = ct.sin(beta)
    cosB = ct.cos(beta)
    cotB = 1 / ct.tan(beta)
    z1 = y1*cosB + a*sinB
    z3 = y1*sinB - a*cosB
    r = ct.sqrt(y1*y1 + y2*y2 + a*a)
    c = 1 - 2*nu
    ra = r + a
    rz = r - z3
    lra = ct.log(ra)
    lrz = ct.log(rz)

    # the Burgers function
    Fi = 2*ct.atan2(y2, ra/ct.tan(beta/2) - y1)

    v1b1 = b[0]/2/pi*((1 - c*cotB*cotB)*Fi + y2/ra*(c*(cotB + y1/2/ra) - y1/r)
                      - y2*(r*sinB - y1)*cosB/r/rz)
    v2b1 = b[0]/2/pi*(c*((0.5 + cotB*cotB)*lra - cotB/sinB*lrz)
                      - 1/ra*(c*(y1*cotB - a/2 - y2*y2/2/ra) + y2*y2/r)
                      + y2*y2*cosB/r/rz)
    v3b1 = b[0]/2/pi*(c*Fi*cotB + y2/ra*(2*nu + a/r) - y2*cosB/rz*(cosB + a/r))

    v1b2 = b[1]/2/pi*(-c*((0.5 - cotB*cotB)*lra + cotB*cotB*cosB*lrz)
                      - 1/ra*(c*(y1*cotB + a/2 + y1*y1/2/ra) - y1*y1/r)
                      + z1*(r*sinB - y1)/r/rz)
    v2b2 = b[1]/2/pi*((1 + c*cotB*cotB)*Fi - y2/ra*(c*(cotB + y1/2/ra) - y1/r)
                      - y2*z1/r/rz)
    v3b2 = b[1]/2/pi*(-c*cotB*(lra - cosB*lrz) - y1/ra*(2*nu + a/r) + z1/rz*(cosB + a/r))

    v1b3 = b[2]/2/pi*(y2*(r*sinB - y1)*sinB/r/rz)
    v2b3 = b[2]/2/pi*(-y2*y2*sinB/r/rz)
    v3b3 = b[2]/2/pi*(Fi + y2*(r*cosB + a)*sinB/r/rz)

    return (v1b1 + v1b2 + v1b3, v2b1 + v2b2 + v2b3, v3b1 + v3b2 + v3b3)


def AngSetupFSC(x, y, b, PA, PB, nu, constants):
    """
    The surface displacements at (x, y) of the angular dislocation pair on the side PA-PB
    """
    pi = tile.constant(constants, PI)
    eps = tile.constant(constants, EPS)

    side = sub(PB, PA)
    cosBeta = -side[2] / norm(side)
    beta = ct.atan2(ct.sqrt(ct.maximum(1 - cosBeta*cosBeta, 0)), cosBeta)
    # a vertical side contributes nothing
    vertical = (ct.abs(beta) < eps) | (ct.abs(pi - beta) < eps)

    # the angular dislocation coordinate system: ey1 is the horizontal direction of the side,
    # ey2 = ey3 x ey1, ey3 points down
    h = ct.sqrt(side[0]*side[0] + side[1]*side[1])
    e1 = side[0] / h
    e2 = side[1] / h
    # the observation point and the burgers vector in that system
    dx = x - PA[0]
    dy = y - PA[1]
    y1A = e1*dx + e2*dy
    y2A = e2*dx - e1*dy
    y1B = y1A - h
    bADCS = (e1*b[0] + e2*b[1], e2*b[0] - e1*b[1], -b[2])

    # pick the artefact-free configuration for the points near the free surface
    angle = ct.where(beta*y1A >= 0, beta - pi, beta)
    vA = AngDisDispSurf(y1A, y2A, angle, bADCS, nu, -PA[2], constants)
    vB = AngDisDispSurf(y1B, y2A, angle, bADCS, nu, -PB[2], constants)
    v = sub(vB, vA)

    # back to the earth fixed system
    u = (v[0]*e1 + v[1]*e2, v[0]*e2 - v[1]*e1, -v[2])
    return select(vertical, (0.0, 0.0, 0.0), u)


def parameters(theta, layout, bs):
    """
    The source parameters of a block of samples, as (TS, 1) columns
    """
    return tuple(ct.astype(ct.load(theta, index=(bs, column), shape=(TS, 1),
                                   padding_mode=ct.PaddingMode.ZERO), ct.float64)
                 for column in ct.static_iter(layout))


@ct.kernel
def sides(theta, table, constants, layout: ct.Constant[tuple], TS: ct.Constant[int]):
    """
    Fill the side table of a block of samples; walking the sides at run time in
    {displacements}, instead of unrolling all twelve, keeps the kernel small enough to compile
    """
    bs = ct.bid(0)
    p = parameters(theta, layout, bs)
    P, Q, S, axes = source(p, constants)
    opening = p[3]
    planes = (P, Q, S)
    # a rectangle with a vanishing side contributes nothing, and has no normal
    actives = ((axes[1] != 0) & (axes[2] != 0), (axes[0] != 0) & (axes[2] != 0),
               (axes[0] != 0) & (axes[1] != 0))

    def put(k, field, value):
        ct.store(table, index=(k, field, bs), tile=ct.reshape(value, (1, 1, TS)))

    for i in ct.static_iter(range(3)):
        V = planes[i]
        active = actives[i]
        normal = cross(sub(V[1], V[0]), sub(V[3], V[0]))
        b = select(active, scale(opening / norm(normal), normal), (0.0, 0.0, 0.0))
        for j in ct.static_iter(range(4)):
            k = 4*i + j
            PA = V[j]
            PB = V[(j + 1) % 4]
            put(k, PAX, PA[0])
            put(k, PAY, PA[1])
            put(k, PAZ, PA[2])
            put(k, PBX, PB[0])
            put(k, PBY, PB[1])
            put(k, PBZ, PB[2])
            put(k, BX, b[0])
            put(k, BY, b[1])
            put(k, BZ, b[2])
            put(k, ACTIVE, ct.where(active, 1.0, 0.0) + 0*opening)


@ct.kernel
def displacements(theta, table, stations, predicted, constants,
                  TS: ct.Constant[int], TO: ct.Constant[int]):
    """
    Fill a (TS x TO) tile of {predicted} with the LOS displacements, less the dataset offsets,
    of the CDM sources in {theta}, whose sides are in {table}; {stations} is laid out as in
    {CDM.load_geometry}
    """
    bs = ct.bid(0)
    bo = ct.bid(1)
    nu = tile.constant(constants, NU)

    # the stations, as (1, TO) rows
    def station(column):
        values = ct.load(stations, index=(bo, column), shape=(TO, 1),
                         padding_mode=ct.PaddingMode.ZERO)
        return ct.reshape(values, (1, TO))
    x = station(0)
    y = station(1)

    # an entry of the side table, as a (TS, 1) column
    def field(k, f):
        values = ct.load(table, index=(k, f, bs), shape=(1, 1, TS),
                         padding_mode=ct.PaddingMode.ZERO)
        return ct.reshape(values, (TS, 1))

    ue = ct.full((TS, TO), 0, dtype=ct.float64)
    un = ct.full((TS, TO), 0, dtype=ct.float64)
    uv = ct.full((TS, TO), 0, dtype=ct.float64)
    for k in range(SIDES):
        PA = (field(k, PAX), field(k, PAY), field(k, PAZ))
        PB = (field(k, PBX), field(k, PBY), field(k, PBZ))
        b = (field(k, BX), field(k, BY), field(k, BZ))
        active = field(k, ACTIVE) != 0
        u = AngSetupFSC(x, y, b, PA, PB, nu, constants)
        ue = ue + ct.where(active, u[0], 0)
        un = un + ct.where(active, u[1], 0)
        uv = uv + ct.where(active, u[2], 0)
    # project along the LOS
    uLOS = ue*station(2) + un*station(3) + uv*station(4)

    # less the offset of each observation's dataset, gathered from its column in {theta}
    column = station(5)
    rows = ct.reshape(ct.arange(TS, dtype=ct.int32) + bs*TS, (TS, 1))
    columns = ct.astype(ct.maximum(column, 0), ct.int32)
    shifted = ct.broadcast_to(column >= 0, (TS, TO))
    offsets = ct.gather(theta, (rows, columns), mask=shifted, padding_value=0)
    uLOS = uLOS - ct.astype(offsets, ct.float64)

    ct.store(predicted, index=(bs, bo), tile=ct.astype(uLOS, predicted.dtype))


@ct.kernel
def verify(theta, mask, constants, layout: ct.Constant[tuple], TS: ct.Constant[int]):
    """
    Flag in {mask} the samples in a block of {theta} whose source reaches above the free surface
    """
    bs = ct.bid(0)
    p = parameters(theta, layout, bs)
    P, Q, S, axes = source(p, constants)
    ok = ct.reshape(buried((P, Q, S)), (TS,))
    flags = ct.load(mask, index=(bs,), shape=(TS,), padding_mode=ct.PaddingMode.ZERO)
    ct.store(mask, index=(bs,), tile=ct.where(ok, flags, 1))


# declaration
class CUDA:
    """
    The cuda strategy: the forward model of all samples at once, as cuTile kernels
    """


    def initialize(self, model):
        """
        Upload the observation geometry
        """
        self.model = model
        self.layout = tuple(int(column) for column in model.layout)
        # the geometry, the constants and the side table are double precision, like the kernels
        self.stations = altar.cuda.matrix(source=model.stations, dtype="float64")
        eps = numpy.finfo("float64").eps
        self.constants = altar.cuda.vector(
            source=[math.pi, math.pi/180, eps, model.nu], dtype="float64")
        # all done
        return self


    def forward_model_batched(self, theta, prediction, batch):
        """
        Fill the first {batch} rows of {prediction} with the LOS displacements of {theta};
        the rest of the last tile of rows gets filled too
        """
        observations = self.stations.shape[0]
        table = self.side_table(samples=theta.shape[0])
        tile.launch(sides, (tile.blocks(batch, TS),),
                    theta, table, self.constants, self.layout, TS)
        tile.launch(displacements,
                    (tile.blocks(batch, TS), tile.blocks(observations, TO)),
                    theta, table, self.stations, prediction, self.constants, TS, TO)
        # all done
        return self


    def verify(self, theta, mask, batch):
        """
        Flag in {mask} the first {batch} samples of {theta} whose source reaches above the free
        surface; the rest of the last tile of samples gets checked too
        """
        tile.launch(verify, (tile.blocks(batch, TS),),
                    theta, mask, self.constants, self.layout, TS)
        # all done
        return self


    def side_table(self, samples):
        """
        The scratch side table, (sides x fields x samples), made once
        """
        if self.table is None or self.table.shape[2] < samples:
            self.table = altar.cuda.managed(
                shape=(SIDES, FIELDS, samples), cell="float64")
        return self.table


    # private data
    model = None
    table = None # the side table
    layout = None # the columns of the source parameters in {theta}
    stations = None # the observation geometry, on the device
    constants = None # the scalars of the kernels, in the working precision


# end of file
