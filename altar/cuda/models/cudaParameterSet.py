# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-2019 parasim inc
# (c) 2010-2019 california institute of technology
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
    # user configurable state
    count = altar.properties.int(default=1)
    count.doc = "the number of parameters in this set"

    prior = altar.cuda.distributions.distribution()
    prior.doc = "the prior distribution"

    prep = altar.cuda.distributions.distribution(default=None)
    prep.doc = "the distribution to use to initialize this parameter set"
    
    offset = altar.properties.int(default=0)


    def cuInitialize(self, application):
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
        self.prior.cuInitialize(application=application)
        if self.prep is not None:
            self.prep.parameters = count
            self.prep.offset = offset
            self.prep.cuInitialize(application=application)
        else:
            self.prep = self.prior

        # return my parameter count so the next set can be initialized properly
        return count
        
    def cuInitSample(self, theta, batch=None):
        """
        Fill {theta} with an initial random sample from my prior distribution.
        """
        # fill it with random numbers from my {prep} distribution
        self.prep.cuInitSample(theta=θ, batch=batch)
        # all done
        return self

    def cuEvalPrior(self, theta, prior, batch=None):
        """
        Fill {priorLLK} with the log likelihoods of the samples in {theta} in my prior distribution
        """
        # delegate
        self.prior.cuEvalPrior(theta=θ, prior=prior, batch=batch)
        # all done
        return self


    @altar.export
    def cuVerify(self, theta, mask, batch=None):
        """
        Check whether the samples in {step.theta} are consistent with the model requirements and
        update the {mask}, a vector with zeroes for valid samples and non-zero for invalid ones
        """
        # ask it to verify my samples
        self.prior.cuVerify(theta=θ, mask=mask, batch=batch)
        # all done; return the rejection map
        return mask


    # implementation details
    def cuRestrict(self, theta):
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
