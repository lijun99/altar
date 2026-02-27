# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-2021 parasim inc
# (c) 2010-2021 california institute of technology
# all rights reserved
#

# the package
import altar

# get the protocol
from . import distribution
# and my base class
from .Base import Base as base


# declaration
class Constant(base, family="altar.distributions.constant"):
    """
    A deterministic distribution that keeps all parameters at fixed values.
    """

    # user configurable state
    value = altar.properties.float(default=0.0)
    value.doc = "the default fixed value"

    values = altar.properties.list(schema=altar.properties.float(), default=None)
    values.doc = "optional per-parameter fixed values"

    # protocol obligations
    @altar.export
    def initialize(self, rng):
        """
        Initialize with the given random number generator.
        """
        self._values = self.resolveValues(parameters=self.parameters)
        return self

    @altar.export
    def initializeSample(self, theta):
        """
        Fill my portion of {theta} with fixed values.
        """
        θ = self.restrict(theta=theta)
        self.matrix(matrix=θ)
        return self

    @altar.export
    def priorLikelihood(self, theta, likelihood):
        """
        Hard constraints contribute an additive constant to log-prior; no-op here.
        """
        return self

    @altar.export
    def verify(self, theta, mask):
        """
        Enforce fixed values directly instead of rejecting samples.
        """
        θ = self.restrict(theta=theta)
        self.matrix(matrix=θ)
        return mask

    @altar.export
    def sample(self):
        """
        Return the fixed scalar value for single-parameter uses.
        """
        return self.resolveValues(parameters=1)[0]

    @altar.export
    def density(self, x):
        """
        Return a pseudo-density used only as a sentinel.
        """
        return 1.0 if x == self.resolveValues(parameters=1)[0] else 0.0

    @altar.export
    def vector(self, vector):
        """
        Fill {vector} with fixed values.
        """
        values = self.resolveValues(parameters=vector.shape)
        for idx, value in enumerate(values):
            vector[idx] = value
        return vector

    @altar.export
    def matrix(self, matrix):
        """
        Fill {matrix} with fixed values row-wise.
        """
        rows = matrix.rows
        columns = matrix.columns
        values = self.resolveValues(parameters=columns)
        for sample in range(rows):
            for parameter, value in enumerate(values):
                matrix[sample, parameter] = value
        return matrix

    # services
    @altar.export
    def fixedParameterMap(self, offset=0, parameters=None):
        """
        Return ``(global_index, value)`` pairs for constrained parameters.
        """
        values = self.resolveValues(parameters=self.parameters if parameters is None else parameters)
        return tuple((offset + index, value) for index, value in enumerate(values))

    # implementation details
    def resolveValues(self, parameters):
        """
        Resolve the fixed value list for a given parameter count.
        """
        # use cached values when available
        if self._values is not None and len(self._values) == parameters:
            return self._values

        values = self.values
        if values is None:
            resolved = tuple(float(self.value) for _ in range(parameters))
        else:
            resolved = list(values)
            if len(resolved) == 1 and parameters > 1:
                resolved = resolved * parameters
            elif len(resolved) != parameters:
                raise ValueError(
                    f"distribution '{type(self).__name__}': expected either one value or "
                    f"{parameters} values; got {len(resolved)}")
            resolved = tuple(float(value) for value in resolved)

        return resolved

    # private data
    _values = None


# end of file
