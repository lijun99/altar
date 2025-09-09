# -*- python -*-
# -*- coding: utf-8 -*-
#
# lijun zhu (ljzhu@gps.caltech.edu)
#
# (c) 2013-2021 parasim inc
# (c) 2010-2021 california institute of technology
# all rights reserved
#

# get the package
import altar
import altar.cuda
import altar.cuda.ext.cudaaltar as libcudaaltar
from altar.models.seismic.ext import cudaseismic as libcudaseismic

# get the protocol

# and my base class
from .cudaMoment import cudaMoment

# the declaration
class cudaMomentLogit(cudaMoment, family="altar.cuda.distributions.momentlogit"):
    """
    The probability distribution for displacements (D) conforming to a given Moment magnitude scale
    Mw = (log M0 - 9.1)/1.5 (Hiroo Kanamori)
    M0 = Mu A D
    It serves to initialize samples only, with combined gaussian and dirichlet distributions.
    It inherits uniform distribution for verification and density calculations.
    """


    def cuInitSample(self, theta, batch):
        """
        Fill my portion of {theta} with initial random values from my distribution.
        """

        # call super method to get ranged slips
        super().cuInitSample(theta=theta, batch=batch)
        # convert to unbounded sampling parameters
        self.cuToSampling(theta=theta, batch=batch)

        # and return
        return self

    # copy methods from cudaUniformLogit
    def cuVerify(self, theta, mask, batch):
        """
        Check whether my portion of the samples in {theta} are consistent with my constraints, and
        update {mask}, a vector with zeroes for valid samples and non-zero for invalid ones
        Arguments:
            theta cuArray (samples x total_parameters)
        """
        # now unbounded, nothing to do
        # all done; return the rejection map
        return mask
    
    def cuEvalPrior(self, theta, prior, batch):
        """
        Fill my portion of {likelihood} with the likelihoods of the samples in {theta}
        """
        # call cuda c extension
        libcudaaltar.cudaLogistic_logpdf(theta.data, prior.data, batch, self.idx_range)

        return self

    def cuEvalPriorwithPhysical(self, theta, prior, batch):   
        """
        cuda process to computes the extra contributions to prior in terms of physical parameters
        """
        # apply moment magnitude constraint (defined in cudaMoment.py)
        super().cuEvalPriorwithPhysical(theta=theta, prior=prior, batch=batch)

        # all done
        return self


    def cuEvalPriorPhysical(self, theta, prior, batch):
        """
        Fill my portion of {prior} with the prior probabilities of the physical samples in {theta}
        """
        # use the cudaUniform_logpdf for physical parameters
        libcudaaltar.cudaUniform_logpdf(theta.data, prior.data, batch, self.idx_range, self.support)

        # all done
        return self

    def cuToPhysical(self, theta, batch):
        """
        Transform {theta} from (-Infty, Infty) to physical ranged parameters with inverse logit function
        """

        libcudaaltar.cudaUniformLogit_tophysical(theta.data, batch, self.idx_range, self.support)
        # all done
        return self

    def cuToSampling(self, theta, batch):
        """
        Transform {theta} from physical ranged parameters to sampling unbounded parameters with logit function
        """

        libcudaaltar.cudaUniformLogit_tosampling(theta.data, batch, self.idx_range, self.support)
        # all done
        return self

    


# end of file
