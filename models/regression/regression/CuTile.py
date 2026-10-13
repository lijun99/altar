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
import cuda.tile as ct
# the package
import altar
import altar.cuda
from altar.cuda import tile


# the tile shape: samples x observations
TS = 16
TO = 128


def row(vector, obs_tile, TO):
    """
    The tile {obs_tile} of {vector}, as a (1, TO) row
    """
    values = ct.load(vector, index=(obs_tile,), shape=(TO,), padding_mode=ct.PaddingMode.ZERO)
    return ct.reshape(values, (1, TO))


def column(theta, sample_tile, parameter, TS):
    """
    The {parameter} of the samples in the tile {sample_tile} of {theta}, as a (TS, 1) column
    """
    return ct.load(theta, index=(sample_tile, parameter), shape=(TS, 1),
                   padding_mode=ct.PaddingMode.ZERO)


@ct.kernel
def predictions(theta, x, prediction,
                slope: ct.Constant[int], intercept: ct.Constant[int],
                TS: ct.Constant[int], TO: ct.Constant[int]):
    """
    Fill a (TS x TO) tile of {prediction} with slope * x + intercept
    """
    sample_tile = ct.bid(0)
    obs_tile = ct.bid(1)
    # the parameters, as (TS, 1) columns
    a = column(theta, sample_tile, slope, TS)
    b = column(theta, sample_tile, intercept, TS)
    ct.store(prediction, index=(sample_tile, obs_tile), tile=a * row(x, obs_tile, TO) + b)


@ct.kernel
def gradients(theta, x, data, weight, grad,
              slope: ct.Constant[int], intercept: ct.Constant[int],
              TS: ct.Constant[int], TO: ct.Constant[int]):
    """
    Fill a tile of TS rows of {grad} with the gradient of the log likelihood,
    -sum_o weight[o] r[o] dr[o]/dθ, with the residual r = prediction - data
    """
    sample_tile = ct.bid(0)
    a = column(theta, sample_tile, slope, TS)
    b = column(theta, sample_tile, intercept, TS)
    # sum over the tiles of observations; the padding has zero weight
    dslope = ct.zeros((TS, 1), dtype=theta.dtype)
    dintercept = ct.zeros((TS, 1), dtype=theta.dtype)
    for obs_tile in range(ct.num_tiles(x, axis=0, shape=(TO,))):
        xs = row(x, obs_tile, TO)
        w = row(weight, obs_tile, TO) * (a * xs + b - row(data, obs_tile, TO))
        dslope = dslope + ct.sum(w * xs, axis=1, keepdims=True)
        dintercept = dintercept + ct.sum(w, axis=1, keepdims=True)
    ct.store(grad, index=(sample_tile, slope), tile=-dslope)
    ct.store(grad, index=(sample_tile, intercept), tile=-dintercept)


# declaration
class CuTile:
    """
    The cuda strategy, with kernels written in python with cuTile
    """


    def initialize(self, model):
        """
        Upload the x
        """
        self.model = model
        self.x = altar.cuda.vector(source=model.x, dtype=model.precision)
        # all done
        return self


    def forward_model_batched(self, theta, prediction, batch):
        """
        Fill the first {batch} rows of {prediction} with the predicted y of {theta}; the rest
        of the last tile of rows gets filled too
        """
        model = self.model
        tile.launch(predictions, (tile.blocks(batch, TS), tile.blocks(self.x.shape[0], TO)),
                    theta, self.x, prediction, model.slopeIdx, model.interceptIdx, TS, TO)
        # all done
        return self


    def gradient(self, theta, data, weight, gradient, batch):
        """
        Fill the first {batch} rows of {gradient} with the gradient of the log likelihood of
        the {data}, whose observations have the inverse variances {weight}; the rest of the
        last tile of rows gets filled too
        """
        model = self.model
        tile.launch(gradients, (tile.blocks(batch, TS),),
                    theta, self.x, data, weight, gradient,
                    model.slopeIdx, model.interceptIdx, TS, TO)
        # all done
        return self


    # private data
    model = None
    x = None # the x of each observation, on the device


# end of file
