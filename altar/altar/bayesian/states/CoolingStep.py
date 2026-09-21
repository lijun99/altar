# -*- python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
#
# (c) 2013-2021 parasim inc
# (c) 2010-2021 california institute of technology
# all rights reserved
#


# the package
import altar
# my base
from .BayesianState import BayesianState


# declaration
class CoolingStep(BayesianState):
    """
    Encapsulation of the state of the calculation at some particular β value

    Extends {BayesianState} with a sampling-space / physical-space split
    ({theta_sampling} vs {theta}) to support reparameterization.
    """


    # public data
    theta_sampling = None     # a (samples x parameters) matrix in sampling space (phi)
    jacobian = None  # a (samples) vector with the log of the Jacobian determinant (d theta/d phi)

    # reparameterization flag
    has_reparametrization = False  # whether reparameterization is implemented

    # my sample matrix lives in {theta_sampling}, not {theta}
    _theta_field = "theta_sampling"
    _theta_label = "θ_sampling"


    @classmethod
    def alloc(cls, samples, parameters, has_reparametrization=False, beta=0):
        """
        Allocate storage for the parts of a cooling step
        """
        # allocate the initial sample set in sampling space
        theta_sampling = altar.matrix(shape=(samples, parameters)).zero()

        # allocate physical parameters and jacobian only if using reparameterization
        theta = None
        jacobian = None
        if has_reparametrization:
            theta = altar.matrix(shape=(samples, parameters)).zero()
            jacobian = altar.vector(shape=samples).zero()

        # allocate the likelihood vectors
        prior, data, posterior = cls._alloc_likelihoods(samples)

        # build one of my instances and return it
        return cls(beta=beta, theta=theta, theta_sampling=theta_sampling,
                  jacobian=jacobian, likelihoods=(prior, data, posterior),
                  has_reparametrization=has_reparametrization)

    # interface
    def clone(self):
        """
        Make a new step with a duplicate of my state
        """
        # make copies of my state
        beta = self.beta
        theta_sampling = self.theta_sampling.clone()
        likelihoods = self.prior.clone(), self.data.clone(), self.posterior.clone()

        # handle physical parameters and jacobian based on reparameterization flag
        theta = self.theta.clone() if self.has_reparametrization else None
        jacobian = self.jacobian.clone() if self.has_reparametrization else None

        # make one and return it
        return type(self)(beta=beta, theta_sampling=theta_sampling, theta=theta,
                         jacobian=jacobian, likelihoods=likelihoods,
                         has_reparametrization=self.has_reparametrization)

    # meta-methods
    def __init__(self, beta, theta_sampling=None, theta=None, jacobian=None, likelihoods=None, has_reparametrization=False, **kwds):
        # chain up (skip BayesianState.__init__, which expects a plain {theta}; go straight
        # to object.__init__)
        super(BayesianState, self).__init__(**kwds)

        # store the temperature
        self.beta = beta
        # fall back to physical parameters when sampling space is not provided
        if theta_sampling is None and theta is not None:
            theta_sampling = theta
        # store the sample sets
        self.theta_sampling = theta_sampling
        if self.theta_sampling is None:
            raise ValueError("CoolingStep requires theta_sampling or theta")
        # store reparameterization flag
        self.has_reparametrization = has_reparametrization

        # handle physical parameters and jacobian based on reparameterization flag
        if has_reparametrization:
            self.theta = theta if theta is not None else theta_sampling.clone()
            self.jacobian = jacobian if jacobian is not None else altar.vector(shape=theta_sampling.rows).zero()
        else:
            # if no reparameterization, physical parameters are the same as sampling parameters
            self.theta = self.theta_sampling
            self.jacobian = None

        # store the likelihoods
        self.prior, self.data, self.posterior = likelihoods

        # all done
        return


    # implementation details
    def _save_parameter_sets_hdf5(self, psetsgrp, psets):
        """
        Write theta_sampling (and, under reparameterization, the physical theta + jacobian)
        into the "ParameterSets" hdf5 group
        """
        import numpy
        # save reparameterization flag
        psetsgrp.create_dataset('has_reparametrization', data=numpy.array([self.has_reparametrization]))

        if len(psets) == 0:
            # no parameter sets info provided, save both parameter spaces
            psetsgrp.create_dataset('theta_sampling', data=self.theta_sampling.ndarray())
            # save physical parameters and jacobian only if using reparameterization
            if self.has_reparametrization:
                psetsgrp.create_dataset('theta', data=self.theta.ndarray())
                psetsgrp.create_dataset('jacobian', data=self.jacobian.ndarray())
        else:
            # get ndarray reference for sampling parameters
            theta_sampling = self.theta_sampling.ndarray()
            # save sampling parameters for all parameter sets
            for name, pset in psets.items():
                psetsgrp.create_dataset(name+'_sampling', data=theta_sampling[:, pset.offset:pset.offset+pset.count])

            # save physical parameters and jacobian only if using reparameterization
            if self.has_reparametrization:
                theta = self.theta.ndarray()
                for name, pset in psets.items():
                    psetsgrp.create_dataset(name+'_physical',
                                         data=theta[:, pset.offset:pset.offset+pset.count])
                # save jacobian
                psetsgrp.create_dataset('jacobian', data=self.jacobian.ndarray())

    def _record_parameter_sets(self, archiver, psets):
        """
        Write theta_sampling (and, under reparameterization, the physical theta + jacobian)
        to the {archiver}
        """
        import numpy
        archiver.write("ParameterSets/has_reparametrization",
                       numpy.array([self.has_reparametrization]))

        if len(psets) == 0:
            archiver.write("ParameterSets/theta_sampling", self.theta_sampling)
            if self.has_reparametrization:
                archiver.write("ParameterSets/theta",    self.theta)
                archiver.write("ParameterSets/jacobian", self.jacobian)
        else:
            theta_sampling = self.theta_sampling.ndarray()
            for name, pset in psets.items():
                archiver.write(f"ParameterSets/{name}_sampling",
                               theta_sampling[:, pset.offset:pset.offset+pset.count])
            if self.has_reparametrization:
                theta = self.theta.ndarray()
                for name, pset in psets.items():
                    archiver.write(f"ParameterSets/{name}_physical",
                                   theta[:, pset.offset:pset.offset+pset.count])
                archiver.write("ParameterSets/jacobian", self.jacobian)

# end of file
