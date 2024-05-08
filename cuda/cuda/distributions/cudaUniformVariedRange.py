# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-2021 parasim inc
# (c) 2010-2021 california institute of technology
# all rights reserved
#
# Author(s): Lijun Zhu

# get the package
import altar
import altar.cuda.ext.cudaaltar as libcudaaltar

# get the parent class
from .cudaUniform import cudaUniform

# externals
import sys

# the declaration
class cudaUniformVariedRange(cudaUniform, family="altar.cuda.distributions.uniformvariedrange"):
    """
    The cuda uniform probability distribution with range control
    """

    # user configurable state
    support = altar.properties.array(default=(0,1))
    support.doc = "the support interval of the prior distribution"

    lows = altar.properties.array(default=None)
    lows.doc = "the lower range of support for each parameter"

    highs = altar.properties.array(default=None)
    highs.doc = "the upper range of support for each parameter"

    low_min = altar.properties.float(default=None)
    low_min.doc = "the minimum value of the lower range"

    low_max = altar.properties.float(default=None)
    low_max.doc = "the maximum value of the lower range"

    #
    lows_vector = None
    highs_vector = None
    glows_vector = None
    ghighs_vector = None

    def cuInitialize(self, application):
        """
        cuda specific initialization
        """
        # run super class method
        super().cuInitialize(application=application)

        parameters = self.parameters

        self.lows_vector=altar.vector(shape=parameters)
        self.highs_vector=altar.vector(shape=parameters)

        self.low_min = self.low_min or -sys.float_info.max
        self.low_max = self.low_max or sys.float_info.max

        low, high = self.support
        # determine the lower ranges
        if self.lows is None:
            # fill in with support value
            self.lows_vector.fill(low)
        elif len(self.lows) != parameters:
            # if the size doesn't match
            channel = self.error
            raise channel.log("the size of lows doesn't match the number of parameters")
        else:
            # copy the tuple over
            for i in range(parameters):
                self.lows_vector[i] = self.lows[i]

        # determine the upper ranges
        if self.highs is None:
            # fill in with support value
            self.highs_vector.fill(high)
        elif len(self.highs) != self.parameters:
            # if the size doesn't match
            channel = self.error
            raise channel.log("the size of highs doesn't match the number of parameters")
        else:
            # copy the tuple over
            for i in range(parameters):
                self.highs_vector[i] = self.highs[i]

        # create gpu vectors and copy from cpu
        self.glows_vector = altar.cuda.vector(source=self.lows_vector, dtype=self.precision)
        self.ghighs_vector = altar.cuda.vector(source=self.highs_vector, dtype=self.precision)

        self.glows_vector.print()
        self.ghighs_vector.print()


        return self


    def cuInitSample(self, theta, batch):
        """
        Fill my portion of {theta} with initial random values from my distribution.
        """

        # call cuda c extension
        libcudaaltar.cudaUniform_sample_unique(theta.data, batch, self.idx_range,
                                               self.glows_vector.data,
                                               self.ghighs_vector.data)

        # and return
        return self

    def cuVerify(self, theta, mask, batch):
        """
        Check whether my portion of the samples in {theta} are consistent with my constraints, and
        update {mask}, a vector with zeroes for valid samples and non-zero for invalid ones
        Arguments:
            theta cuArray (samples x total_parameters)
        """

        # call cuda c extension
        libcudaaltar.cudaRanged_verify_unique(theta.data, mask.data, batch, self.idx_range,
                                              self.glows_vector.data,
                                              self.ghighs_vector.data)

        # all done; return the rejection map
        return mask

    def cuEvalPrior(self, theta, prior, batch):
        """
        Fill my portion of {likelihood} with the likelihoods of the samples in {theta}
        """
        # call cuda c extension
        libcudaaltar.cudaUniform_logpdf_unique(theta.data, prior.data, batch, self.idx_range,
                                               self.glows_vector.data,
                                               self.ghighs_vector.data)

        # all done
        return self

    def update(self, **kwargs):
        """
        Update the support from {std} in {args}
        """

        # grab the std from kwargs
        std = kwargs.get('std')

        # we only update the lower ranges here
        std_max = 0.0
        idx_start, idx_end = self.idx_range

        # change here whether to keep min or max values
        for i in range(self.parameters):
            self.lows_vector[i] = max(self.lows_vector[i], -std[i+idx_start], self.low_min)
            self.lows_vector[i] = min(self.lows_vector[i], self.low_max)

        # copy to gpu vector
        self.glows_vector.copy_from_host(source=self.lows_vector)

        # print(f"new lows")
        # self.glows_vector.print()

        # all done
        return self


    # local variables

# end of file
