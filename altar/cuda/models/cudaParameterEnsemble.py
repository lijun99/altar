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
# the protocol
from .cudaParameter import cudaParameter

# component
class cudaParameterEnsemble(cudaParameter, family="altar.models.parameters.cudaensemble"):
    """
    An Ensemble of parameter sets
    """

    psets = altar.properties.dict(schema=altar.cuda.models.parameters())
    psets.doc = "an ensemble of parameter sets in the model" 

    # interface
    @altar.export
    def initialize(self, application):
        """
        Initialize my distributions
        """
        count = self.cuInitialize(application=application)
        return count


    def cuInitialize(self, application):
        """
        cuda initialize
        """
        # get the parameter sets
        psets = self.psets
        # initialize the offset
        parameters = 0
        # go through my parameter sets
        for name, pset in psets.items():
            # initialize the parameter set
            parameters += pset.cuInitialize(application=application, offset=pset.offset)
        # the total number of parameters is now known, so record it
        self.parameters = parameters

        # return my parameter count so the next set can be initialized properly
        return parameters
        

    def cuInitSample(self, theta, batch=None):
        """
        Fill {theta} with an initial random sample from my prior distribution.
        """
        # ask my subsets
        for pset in self.psets.values():
            # and ask each one to verify the sample
            pset.prep.cuInitSample(theta=θ, batch=batch)

        # all done
        return self



    def cuEvalPrior(self, theta, prior, batch=None):
        """
        Fill {priorLLK} with the log likelihoods of the samples in {theta} in my prior distribution
        """
        # ask my subsets
        for pset in self.psets.values():
            # and ask each one to verify the sample
            pset.prior.cuEvalPrior(theta=θ, prior=prior, batch=batch)

        # all done
        return self


    def cuVerify(self, theta, mask, batch=None):
        """
        Check whether the samples in {step.theta} are consistent with the model requirements and
        update the {mask}, a vector with zeroes for valid samples and non-zero for invalid ones
        """
        # ask my subsets
        for pset in self.psets.values():
            # and ask each one to verify the sample
            pset.prior.cuVerify(theta=θ, mask=mask, batch=batch)
        # all done; return the rejection map
        return mask

# end of file
