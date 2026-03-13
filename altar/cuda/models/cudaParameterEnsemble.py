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
        count = self.cu_initialize(application=application)
        return count


    def cu_initialize(self, application):
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
            parameters += pset.cu_initialize(application=application)
        # the total number of parameters is now known, so record it
        self.count = parameters

        # return my parameter count so the next set can be initialized properly
        return parameters


    def cu_init_sample(self, theta, batch):
        """
        Fill {theta} with an initial random sample from my prior distribution.
        """
        # ask my subsets
        for pset in self._iter_psets():
            # and ask each one to verify the sample
            pset.prep.cu_init_sample(theta=theta, batch=batch)

        # all done
        return self



    def cu_eval_prior(self, theta, prior, batch):
        """
        Fill {prior} with the log likelihoods of the samples in {theta} in my prior distribution
        """
        # ask my subsets
        for pset in self._iter_psets():
            # and ask each one to verify the sample
            pset.prior.cu_eval_prior(theta=theta, prior=prior, batch=batch)

        # all done
        return self


    def cu_verify(self, theta, mask, batch):
        """
        Check whether the samples in {step.theta} are consistent with the model requirements and
        update the {mask}, a vector with zeroes for valid samples and non-zero for invalid ones
        """
        # ask my subsets
        for pset in self._iter_psets():
            # and ask each one to verify the sample
            pset.prior.cu_verify(theta=theta, mask=mask, batch=batch)
        # all done; return the rejection map
        return mask

    def cu_constrain(self, theta, batch):
        """
        Constrain samples to valid parameter space.
        """
        for pset in self._iter_psets():
            pset.prior.cu_constrain(theta=theta, batch=batch)
        return self

    def cu_eval_prior_with_physical(self, theta, prior, batch):
        """
        Compute additional prior contributions in terms of physical parameters.
        """
        for pset in self._iter_psets():
            pset.prior.cu_eval_prior_with_physical(theta=theta, prior=prior, batch=batch)
        return self

    def cu_eval_prior_physical(self, theta, prior, batch):
        """
        Fill {prior} with the log likelihoods of the samples in {theta} in my prior distribution.
        """
        for pset in self._iter_psets():
            pset.prior.cu_eval_prior_physical(theta=theta, prior=prior, batch=batch)
        return self

    def cu_prior_gradient(self, theta, prior, batch):
        """
        Fill {prior} with the log pdf gradient contributions.
        """
        for pset in self._iter_psets():
            pset.prior.cu_prior_gradient(theta=theta, prior=prior, batch=batch)
        return self

# end of file
