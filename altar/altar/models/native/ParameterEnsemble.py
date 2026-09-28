# -*- python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# and my base class
from .Base import Base as base


# the declaration
class ParameterEnsemble(base):
    """
    The cpu implementation of an ensemble of parameter sets
    """


    def initialize(self, model, offset, application=None):
        """
        Initialize my parameter sets given the {model} that owns me
        """
        # set my offset
        self.offset = offset

        # initialize the parameter sets in order
        parameters_total = offset
        for pset in self._iter_psets():
            parameters_total += pset.initialize(model=model, offset=parameters_total)

        # record the count
        self.count = parameters_total - offset

        # return my parameter count so the next set can be initialized properly
        return self.count


    def initialize_sample(self, theta, batch=None):
        """
        Fill {theta} with an initial random sample from my prior distribution.
        """
        for pset in self._iter_psets():
            pset.initialize_sample(theta=theta)
        # all done
        return self


    def eval_prior(self, theta, prior, batch=None):
        """
        Fill {prior} with the log likelihoods of the samples in {theta} in my prior distribution
        """
        for pset in self._iter_psets():
            pset.eval_prior(theta=theta, prior=prior)
        # all done
        return self


    def verify(self, theta, mask, batch=None):
        """
        Check whether the samples in {theta} are consistent with the model requirements and
        update {mask}
        """
        for pset in self._iter_psets():
            pset.verify(theta=theta, mask=mask)
        # all done
        return mask


    def eval_prior_with_physical(self, theta, prior, batch=None):
        """
        Add any prior contributions that depend on physical parameters, e.g. the log-jacobian
        of a reparameterized prior
        """
        for pset in self._iter_psets():
            pset.eval_prior_with_physical(theta=theta, prior=prior, batch=batch)
        return self


    def jacobian(self, theta, jacobian, batch=None):
        """
        Fill {jacobian} with d(physical)/d(sampling), for the reparameterized parameter sets
        """
        for pset in self._iter_psets():
            pset.jacobian(theta=theta, jacobian=jacobian, batch=batch)
        return self


    def to_physical(self, theta, batch=None):
        """
        Transform {theta} from sampling space to physical space, in place
        """
        for pset in self._iter_psets():
            pset.to_physical(theta=theta, batch=batch)
        return self


    def to_sampling(self, theta, batch=None):
        """
        Transform {theta} from physical space to sampling space, in place
        """
        for pset in self._iter_psets():
            pset.to_sampling(theta=theta, batch=batch)
        return self


    # implementation details
    def _iter_psets(self):
        """
        Iterate over parameter sets in a stable order.
        """
        if self.psets_list:
            for name in self.psets_list:
                yield self.psets[name]
        else:
            yield from self.psets.values()


    # private data, set by the shim before {initialize} runs
    psets = None
    psets_list = None


# end of file
