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
from altar.models.ParameterEnsemble import ParameterEnsemble
from altar.models.ParameterSet import ParameterSet as parameters

# component
class cudaParameterEnsemble(ParameterEnsemble,
                            family="altar.cuda.models.parameters.ensemble",
                            implements=parameters):
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
        for pset in self._iter_psets():
            # initialize the parameter set
            pset.offset = parameters
            parameters += pset.cuInitialize(application=application)
        # the total number of parameters is now known, so record it
        self.count = parameters

        # return my parameter count so the next set can be initialized properly
        return parameters


    def cuInitSample(self, theta, batch):
        """
        Fill {theta} with an initial random sample from my prior distribution.
        """
        # ask my subsets
        for pset in self._iter_psets():
            # and ask each one to verify the sample
            pset.prep.cuInitSample(theta=theta, batch=batch)

        # all done
        return self



    def cuEvalPrior(self, theta, prior, batch):
        """
        Fill {priorLLK} with the log likelihoods of the samples in {theta} in my prior distribution
        """
        # ask my subsets
        for pset in self._iter_psets():
            # and ask each one to verify the sample
            pset.prior.cuEvalPrior(theta=theta, prior=prior, batch=batch)

        # all done
        return self


    def cuVerify(self, theta, mask, batch):
        """
        Check whether the samples in {step.theta} are consistent with the model requirements and
        update the {mask}, a vector with zeroes for valid samples and non-zero for invalid ones
        """
        # ask my subsets
        for pset in self._iter_psets():
            # and ask each one to verify the sample
            pset.prior.cuVerify(theta=theta, mask=mask, batch=batch)
        # all done; return the rejection map
        return mask

    def cuConstrain(self, theta, batch):
        """
        Constrain samples to valid parameter space.
        """
        for pset in self._iter_psets():
            pset.prior.cuConstrain(theta=theta, batch=batch)
        return self

    def cuEvalPriorwithPhysical(self, theta, prior, batch):
        """
        Compute additional prior contributions in terms of physical parameters.
        """
        for pset in self._iter_psets():
            pset.prior.cuEvalPriorwithPhysical(theta=theta, prior=prior, batch=batch)
        return self

    def cuEvalPriorPhysical(self, theta, prior, batch):
        """
        Fill {prior} with the log likelihoods of the samples in {theta} in my prior distribution.
        """
        for pset in self._iter_psets():
            pset.prior.cuEvalPriorPhysical(theta=theta, prior=prior, batch=batch)
        return self

    def cuPriorGradient(self, theta, prior, batch):
        """
        Fill {prior} with the log pdf gradient contributions.
        """
        for pset in self._iter_psets():
            pset.prior.cuPriorGradient(theta=theta, prior=prior, batch=batch)
        return self

# end of file
