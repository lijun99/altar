# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# externals
import os
import numpy
# my base class
from .Base import Base as base


# the declaration
class Preset(base):
    """
    The cpu implementation of preset samples; see {altar.distributions.Preset}
    """


    def initialize(self, rng, application=None):
        """
        Find my file and read my samples
        """
        self.samples = load(distribution=self, application=application)
        self.rank = rank(application)
        return self


    def initialize_sample(self, theta, batch=None):
        """
        Fill my portion of {theta} with my samples, this worker's share of them
        """
        θ = numpy.asarray(self.restrict(theta=theta))
        θ[:, :] = rows(samples=self.samples, count=θ.shape[0], rank=self.rank)
        return self


    def verify(self, theta, mask, batch=None):
        """
        I am not a prior: nothing to check
        """
        return mask


    # private data, set by the shim before {initialize} runs
    input_file = None
    dataset = None
    # set by {initialize}
    samples = None
    rank = 0


# helpers shared with the cuda implementation
def load(distribution, application):
    """
    Read the (samples x parameters) dataset of {distribution}, preferring an archived step's
    physical samples
    """
    import h5py
    if distribution.input_file is None or distribution.dataset is None:
        raise ValueError("preset: both 'input_file' and 'dataset' are required")
    path = str(distribution.input_file)
    if not os.path.exists(path) and application is not None:
        path = application.pfs["inputs"][path].uri.path
    with h5py.File(path, "r") as h5:
        name = distribution.dataset
        for candidate in (f"{name}_physical", f"{name}_sampling", name):
            if candidate in h5:
                samples = numpy.asarray(h5[candidate], dtype=float)
                break
        else:
            raise KeyError(f"preset: no dataset '{name}' in '{path}'")
    samples = samples.reshape(samples.shape[0], -1)
    if samples.shape[1] != distribution.parameters:
        raise ValueError(f"preset: '{name}' in '{path}' has {samples.shape[1]} parameters, "
                         f"expected {distribution.parameters}")
    return samples


def rank(application):
    """
    My worker's rank, so that each worker under mpi starts from its own samples
    """
    if application is None:
        return 0
    worker = application.controller.worker
    return getattr(worker, "rank", getattr(worker, "wid", 0))


def rows(samples, count, rank):
    """
    {count} rows of {samples} for the worker of {rank}, wrapping around when there are too few
    """
    return samples[(rank * count + numpy.arange(count)) % samples.shape[0]]


# end of file
