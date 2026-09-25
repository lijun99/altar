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
    The cuda implementation of an ensemble of parameter sets
    """


    def initialize(self, model, offset, application=None):
        """
        Initialize my parameter sets given the current {application}; {model} is unused on
        cuda
        """
        # set my offset
        self.offset = offset

        # initialize the parameter sets in order
        parameters_total = offset
        for pset in self._iter_psets():
            parameters_total += pset.initialize(model=model, offset=parameters_total, application=application)

        # record the count
        self.count = parameters_total - offset

        # return my parameter count so the next set can be initialized properly
        return self.count


    def initialize_sample(self, theta, batch=None):
        """
        Fill {theta} with an initial random sample from my prior distribution.
        """
        for pset in self._iter_psets():
            pset.prep.initialize_sample(theta=theta, batch=batch)
        return self


    def eval_prior(self, theta, prior, batch=None):
        """
        Fill {prior} with the log likelihoods of the samples in {theta} in my prior distribution
        """
        for pset in self._iter_psets():
            pset.prior.eval_prior(theta=theta, likelihood=prior, batch=batch)
        return self


    def verify(self, theta, mask, batch=None):
        """
        Check whether the samples in {theta} are consistent with the model requirements and
        update {mask}
        """
        for pset in self._iter_psets():
            pset.prior.verify(theta=theta, mask=mask, batch=batch)
        return mask


    def constrain(self, theta, batch=None):
        """
        Force the samples in {theta} back within my constraints, in place
        """
        for pset in self._iter_psets():
            pset.prior.constrain(theta=theta, batch=batch)
        return self


    def eval_prior_with_physical(self, theta, prior, batch=None):
        """
        Add any prior contributions that depend on physical parameters, beyond what
        {eval_prior} already contributed
        """
        for pset in self._iter_psets():
            pset.prior.eval_prior_with_physical(theta=theta, likelihood=prior, batch=batch)
        return self


    def eval_prior_physical(self, theta, prior, batch=None):
        """
        Fill {prior} with the log likelihoods of the samples in {theta}, given in physical space
        """
        for pset in self._iter_psets():
            pset.prior.eval_prior_physical(theta=theta, likelihood=prior, batch=batch)
        return self


    def prior_gradient(self, theta, gradient, batch=None):
        r"""
        Fill {gradient} with d\log P(\theta)/d\theta for the samples in {theta}
        """
        for pset in self._iter_psets():
            pset.prior.prior_gradient(theta=theta, gradient=gradient, batch=batch)
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
