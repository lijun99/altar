# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-2021 parasim inc
# (c) 2010-2021 california institute of technology
# all rights reserved
#
# Author(s): Lijun Zhu

# the package
import altar
import altar.cuda
# the base
from altar.models.Contiguous import Contiguous


# component
class cudaParameterSet(Contiguous, family="altar.cuda.models.parameters.parameterset"):
    """
    A contiguous parameter set
    """

    # user configurable state
    count = altar.properties.int(default=1)
    count.doc = "the number of parameters in this set"

    prior = altar.cuda.distributions.distribution()
    prior.doc = "the prior distribution"

    prep = altar.cuda.distributions.distribution(default=None)
    prep.doc = "the distribution to use to initialize this parameter set"

    # parameter set offset in theta
    # determined by cudaBayesian.psets
    offset = 0

    def cu_initialize(self, application):
        """
        cuda initialization
        """
        # get my offset
        offset = self.offset

        # get my count
        count = self.count
        # adjust the number of parameters of my distributions
        self.prior.parameters = count
        self.prior.offset = offset

        # initialize my distributions
        self.prior.cu_initialize(application=application)
        if self.prep is not None:
            self.prep.parameters = count
            self.prep.offset = offset
            self.prep.cu_initialize(application=application)
        else:
            self.prep = self.prior

        # return my parameter count so the next set can be initialized properly
        return count

    def cu_init_sample(self, theta, batch):
        """
        Fill {theta} with an initial random sample from my prior distribution.
        """
        # fill it with random numbers from my {prep} distribution
        self.prep.cu_init_sample(theta=theta, batch=batch)
        # all done
        return self

    def cu_eval_prior(self, theta, prior, batch):
        """
        Fill {prior} with the log likelihoods of the samples in {theta} in my prior distribution
        """
        # delegate
        self.prior.cu_eval_prior(theta=theta, prior=prior, batch=batch)
        # all done
        return self


    @altar.export
    def cu_verify(self, theta, mask, batch):
        """
        Check whether the samples in {step.theta} are consistent with the model requirements and
        update the {mask}, a vector with zeroes for valid samples and non-zero for invalid ones
        """
        # ask it to verify my samples
        self.prior.cu_verify(theta=theta, mask=mask, batch=batch)
        # all done; return the rejection map
        return mask


    # implementation details
    def cu_restrict(self, theta):
        """
        Return my portion of the sample matrix {theta}
        """
        # find out how many samples in the set
        samples = theta.shape[0]
        # get my parameter count
        parameters = self.count
        # get my offset in the samples
        offset = self.offset

        # find where my samples live within the overall sample matrix:
        start = 0, offset
        # form the shape of the sample matrix that's mine
        shape = samples, parameters

        # return a view to the portion of the sample that's mine: i own data in all sample
        # rows, starting in the column indicated by my {offset}, and the width of my block is
        # determined by my parameter count
        return theta.submatrix(start=start, size=shape)



# end of file
