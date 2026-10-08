# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved


# externals
import math
import cuda.tile as ct
# the package
import altar
import altar.cuda
from altar.cuda import tile


# the tile shape: samples x stations
TS = 16
TO = 64
# the entries of the kernel {constants}
K_PI, K_G, K_V, K_MU, K_DRHO, K_GRAVITY, K_GAMMA_S, K_GAMMA_D = range(8)


# the two magma chamber model, as in {lib/libreverso/reverso.cc}; model quantities are (TS, 1)
# tiles, quantities at the stations are (TS, TO) tiles; always in double precision, whatever
# the precision {theta} and {predicted} are stored in

def response(sill, r, H, a, constants):
    """
    The surface displacement, radial and up, at distance {r} per unit overpressure of a chamber
    of radius {a} at depth {H}
    """
    pi = tile.constant(constants, K_PI)
    R2 = r*r + H*H
    f = a*a*a * (1 - tile.constant(constants, K_V)) / (tile.constant(constants, K_G) * R2 * ct.sqrt(R2))
    if sill:
        f = f * 4*H*H / (pi*R2)
    return r*f, H*f


@ct.kernel
def displacements(theta, stations, predicted, constants, layout: ct.Constant[tuple],
                  shallowSill: ct.Constant[bool], deepSill: ct.Constant[bool],
                  TS: ct.Constant[int], TO: ct.Constant[int]):
    """
    Fill the (east, north, up) displacements at a block of TO stations of a block of TS samples
    of {theta} into their columns of {predicted}
    """
    bs = ct.bid(0)
    bo = ct.bid(1)
    pi = tile.constant(constants, K_PI)
    G = tile.constant(constants, K_G)
    mu = tile.constant(constants, K_MU)
    gamma_s = tile.constant(constants, K_GAMMA_S)
    gamma_d = tile.constant(constants, K_GAMMA_D)

    # the models, as (TS, 1) columns
    Qin, H_s, H_d, a_s, a_d, a_c = tuple(
        ct.astype(ct.load(theta, index=(bs, column), shape=(TS, 1),
                          padding_mode=ct.PaddingMode.ZERO), ct.float64)
        for column in ct.static_iter(layout))
    ratio = a_d / a_s
    k = ratio*ratio*ratio
    H_c = H_d - H_s
    gamma_r = gamma_s + gamma_d*k
    a_s3 = a_s*a_s*a_s
    a_c4 = a_c*a_c*a_c*a_c
    # the characteristic time (eq. 10), and the amplitude of the transient
    tau = 8 * mu * H_c * gamma_s * gamma_d * k * a_s3 / (G * a_c4 * gamma_r)
    A = gamma_d*k / gamma_r * (tile.constant(constants, K_DRHO) * tile.constant(constants, K_GRAVITY) * H_c
                               - 8*gamma_s*mu*Qin*H_c / (pi * a_c4 * gamma_r))

    # the stations, as (1, TO) rows
    def station(column):
        values = ct.load(stations, index=(bo, column), shape=(TO, 1),
                         padding_mode=ct.PaddingMode.ZERO)
        return ct.reshape(values, (1, TO))
    t = station(0)
    x = station(1)
    y = station(2)

    # the overpressures
    f0 = A * (1 - ct.exp(-t/tau))
    f1 = G * Qin * t / (pi * a_s3 * gamma_r)
    dP_s = f1 + f0
    dP_d = f1 - f0 * gamma_s / (gamma_d*k)
    # the displacements
    r = ct.sqrt(x*x + y*y)
    ur_s, uz_s = response(shallowSill, r, H_s, a_s, constants)
    ur_d, uz_d = response(deepSill, r, H_d, a_d, constants)
    ur = ur_s*dP_s + ur_d*dP_d
    uz = uz_s*dP_s + uz_d*dP_d
    phi = ct.atan2(y, x)

    # into columns 3 station + component
    rows = ct.reshape(ct.arange(TS, dtype=ct.int32) + bs*TS, (TS, 1))
    columns = ct.reshape(3 * (ct.arange(TO, dtype=ct.int32) + bo*TO), (1, TO))
    dtype = predicted.dtype
    ct.scatter(predicted, (rows, columns), ct.astype(ur * ct.cos(phi), dtype))
    ct.scatter(predicted, (rows, columns + 1), ct.astype(ur * ct.sin(phi), dtype))
    ct.scatter(predicted, (rows, columns + 2), ct.astype(uz, dtype))


@ct.kernel
def verify(theta, mask, layout: ct.Constant[tuple], TS: ct.Constant[int]):
    """
    Flag in {mask} the samples in a block of {theta} whose deep chamber isn't below the shallow one
    """
    bs = ct.bid(0)
    def column(index):
        values = ct.load(theta, index=(bs, index), shape=(TS, 1), padding_mode=ct.PaddingMode.ZERO)
        return ct.reshape(values, (TS,))
    H_s = column(layout[1])
    H_d = column(layout[2])
    flags = ct.load(mask, index=(bs,), shape=(TS,), padding_mode=ct.PaddingMode.ZERO)
    ct.store(mask, index=(bs,), tile=ct.where(H_d > H_s, flags, 1))


# declaration
class CUDA:
    """
    The cuda strategy: the forward model of all samples at once, as a cuTile kernel
    """


    def initialize(self, model):
        """
        Upload the observation geometry
        """
        self.model = model
        self.layout = tuple(int(column) for column in model.layout)
        # the geometry and the constants are double precision, like the kernel
        self.stations = altar.cuda.matrix(source=model.stations, dtype="float64")
        sill = 8 * (1 - model.v) / (3 * math.pi)
        self.constants = altar.cuda.vector(
            source=[math.pi, model.G, model.v, model.mu, model.drho, model.g,
                    sill if model.shallow == "sill" else 1.0,
                    sill if model.deep == "sill" else 1.0],
            dtype="float64")
        # all done
        return self


    def forward_model_batched(self, theta, prediction, batch):
        """
        Fill the first {batch} rows of {prediction} with the displacements of {theta}; the rest
        of the last tile of rows gets filled too
        """
        model = self.model
        stations = self.stations.shape[0]
        tile.launch(displacements, (tile.blocks(batch, TS), tile.blocks(stations, TO)),
                    theta, self.stations, prediction, self.constants, self.layout,
                    model.shallow == "sill", model.deep == "sill", TS, TO)
        # all done
        return self


    def verify(self, theta, mask, batch):
        """
        Flag in {mask} the first {batch} samples of {theta} whose deep chamber isn't below the
        shallow one; the rest of the last tile of samples gets checked too
        """
        tile.launch(verify, (tile.blocks(batch, TS),), theta, mask, self.layout, TS)
        # all done
        return self


    # private data
    model = None
    layout = None # the columns of the model parameters in {theta}
    stations = None # the observation geometry, on the device
    constants = None # the scalars of the kernel, in double precision


# end of file
