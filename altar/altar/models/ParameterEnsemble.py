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
# the protocol
from .ParameterSet import ParameterSet as parameters


# component
class ParameterEnsemble(altar.component,
                        family="altar.models.parameters.parameterensemble",
                        implements=parameters):
    """
    An ensemble of parameter sets
    """

    # user configurable state
    count = altar.properties.int(default=0)
    count.doc = "the total number of parameters in this ensemble"

    prior = altar.distributions.distribution(default=None)
    prior.doc = "not used; provided for protocol compatibility"

    prep = altar.distributions.distribution(default=None)
    prep.doc = "not used; provided for protocol compatibility"

    psets_list = altar.properties.list(schema=altar.properties.str(), default=None)
    psets_list.doc = "list of parameter set names that defines the order"

    psets = altar.properties.dict(schema=parameters())
    psets.default = dict()
    psets.doc = "the collection of parameter sets in the ensemble"

    # state set by the model
    offset = 0


    # interface
    @altar.export
    def initialize(self, model, offset):
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


    @altar.export
    def initialize_sample(self, theta):
        """
        Fill {theta} with an initial random sample from my prior distribution.
        """
        for pset in self._iter_psets():
            pset.initialize_sample(theta=theta)
        # all done
        return self


    @altar.export
    def eval_prior(self, theta, prior):
        """
        Fill {prior} with the log likelihoods of the samples in {theta} in my prior distribution
        """
        for pset in self._iter_psets():
            pset.eval_prior(theta=theta, prior=prior)
        # all done
        return self


    @altar.export
    def verify(self, theta, mask):
        """
        Check whether the samples in {theta} are consistent with the model requirements and
        update {mask}
        """
        for pset in self._iter_psets():
            pset.verify(theta=theta, mask=mask)
        # all done
        return mask


    # implementation details
    def restrict(self, theta):
        """
        Return my portion of the sample matrix {theta}
        """
        # find out how many samples in the set
        samples = theta.rows
        # get my parameter count
        parameters = self.count
        # get my offset in the samples
        offset = self.offset

        # find where my samples live within the overall sample matrix
        start = 0, offset
        # form the shape of the sample matrix that's mine
        shape = samples, parameters

        # return a view to the portion of the sample that's mine
        return theta.view(start=start, shape=shape)


    def _iter_psets(self):
        """
        Iterate over parameter sets in a stable order.
        """
        if self.psets_list:
            for name in self.psets_list:
                yield self.psets[name]
        else:
            yield from self.psets.values()


# end of file
