# -*- python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# get the package
import altar

# and my base class
from .Base import Base as base


# the declaration
class Uniform(base):
    """
    The cpu implementation of the uniform probability distribution
    """


    def initialize(self, rng, application=None):
        """
        Initialize with the given random number generator
        """
        # set up my pdf
        self.pdf = altar.pdf.uniform(rng=rng.rng, support=self.support)
        # if reparameterizing, hand my transform its bounds and let it initialize
        if self.reparameterize:
            self.has_reparametrization = True
            self.transform.support = self.support
            self.transform.initialize(application=application)
        # all done
        return self


    def verify(self, theta, mask, batch=None):
        """
        Check whether my portion of the samples in {theta} are consistent with my constraints, and
        update {mask}, a vector with zeroes for valid samples and non-zero for invalid ones;
        a no-op when reparameterized, since sampling space is unconstrained
        """
        # reparameterized: sampling space is unbounded, nothing to verify
        if self.reparameterize:
            return mask

        # unpack my support
        low, high = self.support
        # grab the portion of the sample that's mine
        θ = self.restrict(theta=theta)

        # find out how many samples in the set
        samples = θ.rows
        # and how many parameters belong to me
        parameters = θ.columns

        # go through the samples in θ
        for sample in range(samples):
            # and the parameters in this sample
            for parameter in range(parameters):
                # if the parameter lies outside my support
                if not (low <= θ[sample,parameter] <= high):
                    # mark the entire sample as invalid
                    mask[sample] += 1
                    # and skip checking the rest of the parameters
                    break

        # all done; return the rejection map
        return mask


    def to_physical(self, theta, batch=None):
        """
        Transform my portion of {theta} from sampling space to physical space, in place; a
        no-op unless reparameterized
        """
        if self.reparameterize:
            θ = self.restrict(theta=theta)
            self.transform.to_physical(theta=θ, batch=batch)
        return self


    def to_sampling(self, theta, batch=None):
        """
        Transform my portion of {theta} from physical space to sampling space, in place; the
        inverse of {to_physical}, a no-op unless reparameterized
        """
        if self.reparameterize:
            θ = self.restrict(theta=theta)
            self.transform.to_sampling(theta=θ, batch=batch)
        return self


    def eval_prior_with_physical(self, theta, likelihood, batch=None):
        """
        Add the transform's log-jacobian into {likelihood}, when reparameterized
        """
        if self.reparameterize:
            θ = self.restrict(theta=theta)
            self.transform.log_jacobian(theta=θ, likelihood=likelihood, batch=batch)
        return self


    def prior_gradient(self, theta, gradient, batch=None):
        r"""
        Fill my portion of {gradient} with d\log P(\theta)/d\theta; when reparameterized,
        this is exactly the transform's jacobian-gradient, since a uniform prior's
        physical-space gradient is always zero
        """
        g = self.restrict(theta=gradient)
        if self.reparameterize:
            θ = self.restrict(theta=theta)
            self.transform.jacobian_gradient(theta=θ, gradient=g, batch=batch)
        else:
            g.zero()
        return self


    def jacobian(self, theta, jacobian, batch=None):
        """
        Fill my portion of {jacobian} with d(physical)/d(sampling), or 1 when not
        reparameterized
        """
        if self.reparameterize:
            θ = self.restrict(theta=theta)
            j = self.restrict(theta=jacobian)
            self.transform.jacobian(theta=θ, jacobian=j, batch=batch)
        else:
            self.restrict(theta=jacobian).fill(1.0)
        return self


    # private data, set by the shim before {initialize} runs
    support = None
    reparameterize = False
    transform = None


# end of file
