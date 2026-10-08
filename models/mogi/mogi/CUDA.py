# -*- python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#


# externals
import math
import cuda.tile as ct
# the package
import altar
import altar.cuda
from altar.cuda import tile


# the tile shape: samples x observations
TS = 16
TO = 128


@ct.kernel
def displacements(theta, stations, predicted,
                  xIdx: ct.Constant[int], yIdx: ct.Constant[int],
                  dIdx: ct.Constant[int], sIdx: ct.Constant[int],
                  log10dV: ct.Constant[bool], constants,
                  TS: ct.Constant[int], TO: ct.Constant[int]):
    """
    Fill a (TS x TO) tile of {predicted} with the LOS displacements, less the dataset offsets,
    of the Mogi sources in {theta}; {stations} is laid out as in {Mogi.load_geometry}, and
    {constants} holds (1-nu)/pi
    """
    bs = ct.bid(0)
    bo = ct.bid(1)

    # the sources, as (TS, 1) columns
    xs = ct.load(theta, index=(bs, xIdx), shape=(TS, 1), padding_mode=ct.PaddingMode.ZERO)
    ys = ct.load(theta, index=(bs, yIdx), shape=(TS, 1), padding_mode=ct.PaddingMode.ZERO)
    ds = ct.load(theta, index=(bs, dIdx), shape=(TS, 1), padding_mode=ct.PaddingMode.ZERO)
    dV = ct.load(theta, index=(bs, sIdx), shape=(TS, 1), padding_mode=ct.PaddingMode.ZERO)
    if log10dV:
        dV = ct.pow(10.0, dV)

    # the stations, as (1, TO) rows
    def station(column):
        values = ct.load(stations, index=(bo, column), shape=(TO, 1),
                         padding_mode=ct.PaddingMode.ZERO)
        return ct.reshape(values, (1, TO))
    x = station(0) - xs
    y = station(1) - ys
    R2 = x*x + y*y + ds*ds
    C = tile.constant(constants, 0) * dV / (R2 * ct.sqrt(R2))
    u = C * (x*station(2) + y*station(3) + ds*station(4))

    # less the offset of each observation's dataset, gathered from its column in {theta}
    column = station(5)
    rows = ct.reshape(ct.arange(TS, dtype=ct.int32) + bs*TS, (TS, 1))
    columns = ct.astype(ct.maximum(column, 0), ct.int32)
    shifted = ct.broadcast_to(column >= 0, (TS, TO))
    u = u - ct.gather(theta, (rows, columns), mask=shifted, padding_value=0)

    ct.store(predicted, index=(bs, bo), tile=u)


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
        self.stations = altar.cuda.matrix(source=model.stations, dtype=model.precision)
        self.constants = altar.cuda.vector(source=[(1 - model.nu) / math.pi], dtype=model.precision)
        # all done
        return self


    def forward_model_batched(self, theta, prediction, batch):
        """
        Fill the first {batch} rows of {prediction} with the LOS displacements of {theta};
        the rest of the last tile of rows gets filled too
        """
        model = self.model
        observations = self.stations.shape[0]
        tile.launch(displacements,
                    (tile.blocks(batch, TS), tile.blocks(observations, TO)),
                    theta, self.stations, prediction,
                    model.xIdx, model.yIdx, model.dIdx, model.sIdx,
                    model.log10_dV, self.constants, TS, TO)
        # all done
        return self


    # private data
    model = None
    stations = None # the observation geometry, on the device
    constants = None # the scalars of the kernel, in the working precision


# end of file
