# -*- python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#


# externals
from __future__ import annotations
import typing
import numpy
# my base
from .BayesianState import BayesianState, Likelihoods

if typing.TYPE_CHECKING:
    import h5py
    from altar.bayesian.controllers.Annealer import Annealer
    from altar.models.Bayesian import Bayesian
    from altar.simulations.Archiver import Archiver


# declaration
class CoolingStep(BayesianState):
    """
    Encapsulation of the state of the calculation at some particular β value

    Extends {BayesianState} with a sampling-space / physical-space split
    ({theta_sampling} vs {theta}) to support reparameterization.
    """


    # public data
    theta_sampling: numpy.ndarray        # (samples x parameters), in sampling space (phi)
    jacobian: numpy.ndarray | None = None # (samples,) log of the jacobian determinant d theta/d phi

    # reparameterization flag
    has_reparametrization: bool = False  # whether reparameterization is implemented

    # my sample matrix lives in {theta_sampling}, not {theta}
    _theta_field: str = "theta_sampling"
    _theta_label: str = "θ_sampling"


    @classmethod
    def allocate(cls, annealer: Annealer) -> typing.Self:
        """
        Build an uninitialized step sized for {annealer}'s model, carrying the extra
        sampling/physical-space state when the model is reparameterized -- overrides
        {BayesianState.allocate}, which doesn't know about {has_reparametrization} at all
        """
        model = annealer.model
        has_reparametrization = getattr(model, 'has_reparametrization', False)
        return cls.alloc(samples=model.job.chains, parameters=model.parameters,
                         has_reparametrization=has_reparametrization, dtype=model.job.precision)


    @classmethod
    def alloc(cls, samples: int, parameters: int, has_reparametrization: bool = False,
              beta: float = 0, dtype: str = "float64") -> typing.Self:
        """
        Allocate storage for the parts of a cooling step
        """
        # allocate the initial sample set in sampling space
        theta_sampling = numpy.zeros((samples, parameters), dtype=dtype)

        # allocate physical parameters and jacobian only if using reparameterization
        theta = None
        jacobian = None
        if has_reparametrization:
            theta = numpy.zeros((samples, parameters), dtype=dtype)
            jacobian = numpy.zeros(samples)

        # allocate the likelihood vectors
        prior, data, posterior = cls._alloc_likelihoods(samples)

        # build one of my instances and return it
        return cls(beta=beta, theta=theta, theta_sampling=theta_sampling,
                  jacobian=jacobian, likelihoods=(prior, data, posterior),
                  has_reparametrization=has_reparametrization)

    # interface
    def reorder(self, rows: numpy.ndarray) -> typing.Self:
        """
        Replace each chain {i} by the chain {rows[i]}, in place: its samples, in sampling and
        physical space, its jacobian, and its densities
        """
        self.theta_sampling[...] = self.theta_sampling[rows]
        # a reparameterized step keeps its physical samples and jacobian apart
        if self.has_reparametrization:
            self.theta[...] = self.theta[rows]
            self.jacobian[...] = self.jacobian[rows]
        for density in (self.prior, self.data, self.posterior):
            density[...] = density[rows]
        return self


    def clone(self) -> typing.Self:
        """
        Make a new step with a duplicate of my state
        """
        # make copies of my state
        beta = self.beta
        theta_sampling = self.theta_sampling.copy()
        likelihoods = self.prior.copy(), self.data.copy(), self.posterior.copy()

        # handle physical parameters and jacobian based on reparameterization flag
        theta = self.theta.copy() if self.has_reparametrization else None
        jacobian = self.jacobian.copy() if self.has_reparametrization else None

        # make one and return it
        return type(self)(beta=beta, theta_sampling=theta_sampling, theta=theta,
                         jacobian=jacobian, likelihoods=likelihoods,
                         has_reparametrization=self.has_reparametrization)

    # meta-methods
    def __init__(self, beta: float, theta_sampling: numpy.ndarray | None = None,
                 theta: numpy.ndarray | None = None, jacobian: numpy.ndarray | None = None,
                 likelihoods: Likelihoods | None = None, has_reparametrization: bool = False,
                 **kwds) -> None:
        # chain up (skip BayesianState.__init__, which expects a plain {theta}; go straight
        # to object.__init__)
        super(BayesianState, self).__init__(**kwds)

        # store the temperature
        self.beta = beta
        # fall back to physical parameters when sampling space is not provided
        if theta_sampling is None:
            if theta is None:
                raise ValueError("CoolingStep requires theta_sampling or theta")
            theta_sampling = theta
        if likelihoods is None:
            raise ValueError("CoolingStep requires likelihoods")
        # store the sample sets
        self.theta_sampling = theta_sampling
        # store reparameterization flag
        self.has_reparametrization = has_reparametrization

        # handle physical parameters and jacobian based on reparameterization flag
        if has_reparametrization:
            self.theta = theta if theta is not None else theta_sampling.copy()
            self.jacobian = jacobian if jacobian is not None else numpy.zeros(theta_sampling.shape[0])
        else:
            # if no reparameterization, physical parameters are the same as sampling parameters
            self.theta = self.theta_sampling
            self.jacobian = None

        # store the likelihoods
        self.prior, self.data, self.posterior = likelihoods

        # all done
        return


    def refresh_sampling(self, model: Bayesian, batch: int | None = None) -> typing.Self:
        """
        Rebuild {theta_sampling} and {jacobian} from the physical {theta}, e.g. after a walk in
        physical space
        """
        if not self.has_reparametrization:
            return self
        self.theta_sampling[...] = self.theta
        model.to_sampling(theta=self.theta_sampling, batch=batch)
        self.jacobian[:] = 0
        model.eval_prior_with_physical(step=self, likelihood=self.jacobian, batch=batch)
        return self


    # implementation details
    def _on_start(self, annealer: Annealer) -> None:
        """
        The jacobian of the initial samples, when reparameterized
        """
        if self.has_reparametrization:
            self.jacobian[:] = 0
            annealer.model.eval_prior_with_physical(step=self, likelihood=self.jacobian)
        return


    def _save_parameter_sets_hdf5(self, psetsgrp: h5py.Group, psets: dict) -> None:
        """
        Write theta_sampling (and, under reparameterization, the physical theta + jacobian)
        into the "ParameterSets" hdf5 group
        """
        # save reparameterization flag
        psetsgrp.create_dataset('has_reparametrization', data=numpy.array([self.has_reparametrization]))

        if len(psets) == 0:
            # no parameter sets info provided, save both parameter spaces
            psetsgrp.create_dataset('theta_sampling', data=self.theta_sampling)
            # save physical parameters and jacobian only if using reparameterization
            if self.has_reparametrization:
                psetsgrp.create_dataset('theta', data=self.theta)
                psetsgrp.create_dataset('jacobian', data=self.jacobian)
        else:
            theta_sampling = self.theta_sampling
            # save sampling parameters for all parameter sets
            for name, pset in psets.items():
                psetsgrp.create_dataset(name+'_sampling', data=theta_sampling[:, pset.offset:pset.offset+pset.count])

            # save physical parameters and jacobian only if using reparameterization
            if self.has_reparametrization:
                theta = self.theta
                for name, pset in psets.items():
                    psetsgrp.create_dataset(name+'_physical',
                                         data=theta[:, pset.offset:pset.offset+pset.count])
                # save jacobian
                psetsgrp.create_dataset('jacobian', data=self.jacobian)

    def _record_parameter_sets(self, archiver: Archiver, psets: dict) -> None:
        """
        Write theta_sampling (and, under reparameterization, the physical theta + jacobian)
        to the {archiver}
        """
        archiver.write("ParameterSets/has_reparametrization",
                       numpy.array([self.has_reparametrization]))

        if len(psets) == 0:
            archiver.write("ParameterSets/theta_sampling", self.theta_sampling)
            if self.has_reparametrization:
                archiver.write("ParameterSets/theta",    self.theta)
                archiver.write("ParameterSets/jacobian", self.jacobian)
        else:
            theta_sampling = self.theta_sampling
            for name, pset in psets.items():
                archiver.write(f"ParameterSets/{name}_sampling",
                               theta_sampling[:, pset.offset:pset.offset+pset.count])
            if self.has_reparametrization:
                theta = self.theta
                for name, pset in psets.items():
                    archiver.write(f"ParameterSets/{name}_physical",
                                   theta[:, pset.offset:pset.offset+pset.count])
                archiver.write("ParameterSets/jacobian", self.jacobian)

# end of file
