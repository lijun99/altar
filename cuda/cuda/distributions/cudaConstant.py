# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-2021 parasim inc
# (c) 2010-2021 california institute of technology
# all rights reserved
#

# the package
import altar
import altar.cuda

# the base
from .cudaDistribution import cudaDistribution


# declaration
class cudaConstant(cudaDistribution, family="altar.cuda.distributions.constant"):
    """
    A deterministic distribution that keeps all parameters at fixed values.
    """

    # user configurable state
    value = altar.properties.float(default=0.0)
    value.doc = "the default fixed value"

    values = altar.properties.list(schema=altar.properties.float(), default=None)
    values.doc = "optional per-parameter fixed values"

    def cuInitialize(self, application):
        """
        cuda specific initialization
        """
        super().cuInitialize(application=application)
        self._values = self.resolveValues(parameters=self.parameters)
        return self

    def cuInitSample(self, theta, batch):
        """
        Fill my portion of {theta} with fixed values.
        """
        import numpy

        values = numpy.asarray(self.resolveValues(parameters=self.parameters), dtype=theta.dtype)
        hmatrix = numpy.tile(values, (batch, 1))
        dmatrix = altar.cuda.matrix(source=hmatrix, dtype=hmatrix.dtype)
        theta.insert(src=dmatrix, start=(0, self.offset))
        return self

    def cuVerify(self, theta, mask, batch):
        """
        Nothing to verify; proposal covariance already keeps constrained dims unchanged.
        """
        return mask

    def cuEvalPrior(self, theta, prior, batch):
        """
        Hard constraints contribute an additive constant to log-prior; no-op here.
        """
        return prior

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
