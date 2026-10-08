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
# the package
import altar

if typing.TYPE_CHECKING:
    import h5py
    import journal
    from altar.bayesian.controllers.Annealer import Annealer
    from altar.simulations.Archiver import Archiver

# the prior, data and posterior log likelihoods of a step
Likelihoods = tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]


# declaration
class BayesianState:
    """
    Encapsulation of the Bayesian state of the calculation

    This is the common base for the state classes used by the various samplers
    (CATMIP/Metropolis's {CoolingStep}, HMC's {HMCState}, ...). It owns the
    parts of the state that are shared by all of them -- the sample matrix, the
    prior/data/posterior likelihood vectors, and the generic
    allocate/clone/persist machinery -- and exposes a handful of override
    seams so subclasses can add their own fields (reparameterization,
    gradients, ...) without duplicating the rest.
    """

    # public data
    beta: float
    theta: numpy.ndarray     # (samples x parameters)
    prior: numpy.ndarray     # (samples,) logs of the prior
    data: numpy.ndarray      # (samples,) logs of the data likelihoods given the samples
    posterior: numpy.ndarray # (samples,) logs of the posterior
    # (samples,) importance weights w_i ∝ exp(Δβ · data_i), set by the scheduler
    weights: numpy.ndarray | None = None

    # the statistics of samples
    mean: numpy.ndarray | None = None
    sd: numpy.ndarray | None = None

    # the name of the attribute that holds the (samples x parameters) sample matrix, and
    # the label used for it in diagnostic output; subclasses whose sample matrix lives
    # under a different name (e.g. {CoolingStep.theta_sampling}) override these
    _theta_field: str = "theta"
    _theta_label: str = "θ"


    # read-only public data
    @property
    def _shape_matrix(self) -> numpy.ndarray:
        """
        The (samples x parameters) matrix that determines my shape
        """
        return getattr(self, self._theta_field)

    @property
    def samples(self) -> int:
        """
        The number of samples
        """
        return self._shape_matrix.shape[0]

    @property
    def parameters(self) -> int:
        """
        The number of model parameters
        """
        return self._shape_matrix.shape[1]


    @classmethod
    def start(cls, annealer: Annealer) -> typing.Self:
        """
        Build the first cooling step by asking {model} to produce a sample set from its
        initializing prior, compute the likelihood of this sample given the data, and compute a
        (perhaps trivial) posterior
        """
        # get the model
        model = annealer.model
        # build an uninitialized step, with the reparameterization state when my class has one
        step = cls.allocate(annealer=annealer)

        # initialize it
        model.initialize_sample(step=step)
        # compute the likelihoods
        model.likelihoods(annealer=annealer, step=step)
        # let subclasses do any extra work that depends on the likelihoods being ready
        step._on_start(annealer=annealer)

        # return the initialized state
        return step

    @classmethod
    def allocate(cls, annealer: Annealer) -> typing.Self:
        # get the model
        model = annealer.model
        # build an uninitialized step
        step = cls.alloc(samples=model.job.chains, parameters=model.parameters,
                         dtype=model.job.precision)
        return step

    @classmethod
    def alloc(cls, samples: int, parameters: int, dtype: str = "float64") -> typing.Self:
        """
        Allocate storage for the parts of a cooling step: the samples in {dtype}, the log
        densities in double precision
        """
        # allocate the initial sample set
        theta = numpy.zeros((samples, parameters), dtype=dtype)
        # allocate the likelihood vectors
        prior, data, posterior = cls._alloc_likelihoods(samples)
        # build one of my instances and return it
        return cls(beta=0, theta=theta, likelihoods=(prior, data, posterior))

    @classmethod
    def _alloc_likelihoods(cls, samples: int) -> Likelihoods:
        """
        Allocate the (samples) prior/data/posterior likelihood vectors
        """
        prior = numpy.zeros(samples)
        data = numpy.zeros(samples)
        posterior = numpy.zeros(samples)
        return prior, data, posterior

    def _on_start(self, annealer: Annealer) -> None:
        """
        Hook invoked by {start} right after the likelihoods have been computed; the base
        implementation does nothing. {HMCState} uses this to compute gradients.
        """
        return


    # interface
    def clone(self) -> typing.Self:
        """
        Make a new step with a duplicate of my state
        """
        # make copies of my state
        beta = self.beta
        theta = self.theta.copy()
        likelihoods = self.prior.copy(), self.data.copy(), self.posterior.copy()

        # make one and return it
        return type(self)(beta=beta, theta=theta, likelihoods=likelihoods)

    def compute_posterior(self) -> typing.Self:
        """
        Compute the posterior from prior, data, and beta
        """

        # in their log form, posterior = prior + beta * datalikelihood
        self.posterior[:] =self.prior + self.beta * self.data
        # all done
        return self

    def statistics(self) -> typing.Self:
        """
        Compute the statistics of samples
        :return:
        """
        # get the samples
        θ = self._shape_matrix
        # compute the mean, sd
        self.mean, self.sd = θ.mean(axis=0), θ.std(axis=0, ddof=1)
        # all done
        return self

    # meta-methods
    def __init__(self, beta: float, theta: numpy.ndarray, likelihoods: Likelihoods,
                 **kwds) -> None:
        # chain up
        super().__init__(**kwds)

        # store the temperature
        self.beta = beta
        # store the sample set
        self.theta = theta
        # store the likelihoods
        self.prior, self.data, self.posterior = likelihoods

        # all done
        return


    # implementation details
    def print(self, channel: journal.info, indent: str = ' '*2) -> journal.info:
        """
        Print info about this step
        """
        # unpack my shape
        samples = self.samples
        parameters = self.parameters

        # say something
        channel.line(f"step")
        # show me the temperature
        channel.line(f"{indent}β: {self.beta}")
        # the sample
        θ = self._shape_matrix
        channel.line(f"{indent}{self._theta_label}: ({samples} samples) x ({parameters} parameters)")
        if samples <= 10 and parameters <= 10:
            channel.line(indent * 2 + str(θ).replace("\n", "\n" + indent * 2))

        if samples < 10:
            # the likelihoods
            for name in ("prior", "data", "posterior"):
                channel.line(f"{indent}{name}:")
                channel.line(indent * 2 + str(getattr(self, name)))

        # print statistics (axis=0 average over samples)
        mean, sd = self.mean, self.sd
        channel.line(f"{indent}parameters (mean, sd):")
        if parameters <= 25:
            for i in range(parameters):
                channel.line(f"{indent} ({mean[i]}, {sd[i]})")
        else:
            for i in range(20):
                channel.line(f"{indent} ({mean[i]}, {sd[i]})")
            channel.line(f"{indent} ... ...")
            for i in range(parameters-5, parameters):
                channel.line(f"{indent} ({mean[i]}, {sd[i]})")

        # let subclasses add anything extra (e.g. gradient summaries)
        self._extra_print(channel=channel, indent=indent)

        # flush
        channel.log()

        # all done
        return channel

    def _extra_print(self, channel: journal.info, indent: str) -> None:
        """
        Hook for subclasses to print additional state; no-op by default
        """
        return

    def save_hdf5(self, path: str | altar.primitives.path | None = None,
                  iteration: int | None = None, psets: dict | None = None) -> None:
        """
        Save this step to an HDF5 file

        Args:

            path altar.primitives.path
            iteration the iteration number, or None for the final step
            psets the named parameter sets, or an empty dict/None to save the raw matrix

        Returns:

            None
        """
        import os
        import h5py

        # determine the output name as "{path}/step_{iteration}.h5"
        str_iteration = 'final' if iteration is None else str(iteration).zfill(3)
        if path is not None:
            str_path = path.path if isinstance(path, altar.primitives.path) else path
            if not os.path.exists(str_path):
                os.makedirs(str_path)
        else:
            str_path = '.'
        suffix = '.h5'
        filename = os.path.join(str_path, "step_"+str_iteration+suffix)

        # create a hdf5 file
        f=h5py.File(filename, 'w')
        # save annealer info
        annealergrp = f.create_group('Annealer')
        annealergrp.create_dataset('beta', data=numpy.asarray(self.beta))
        # save parameter sets
        psetsgrp = f.create_group('ParameterSets')
        self._save_parameter_sets_hdf5(psetsgrp=psetsgrp, psets=psets or {})
        # save Bayesian likelihoods/probabilities
        bayesiangrp = f.create_group('Bayesian')
        bayesiangrp.create_dataset('prior', data=self.prior)
        bayesiangrp.create_dataset('likelihood', data=self.data)
        bayesiangrp.create_dataset('posterior', data=self.posterior)
        # let subclasses save anything extra (e.g. gradients)
        self._extra_save_hdf5(f=f)
        f.close()

        # all done
        return

    def _save_parameter_sets_hdf5(self, psetsgrp: h5py.Group, psets: dict) -> None:
        """
        Write my sample matrix into the "ParameterSets" hdf5 group; overridden by subclasses
        whose sample matrix needs more than a single flat dataset (e.g. reparameterization)
        """
        theta = self._shape_matrix
        if len(psets) == 0:
            psetsgrp.create_dataset(self._theta_field, data=theta)
        else:
            for name, pset in psets.items():
                psetsgrp.create_dataset(name, data=theta[:, pset.offset:pset.offset+pset.count])

    def _extra_save_hdf5(self, f: h5py.File) -> None:
        """
        Hook for subclasses to persist additional state (e.g. gradients); no-op by default
        """
        return

    def record(self, archiver: Archiver) -> typing.Self:
        """
        Record me using the provided {archiver}.
        """
        psets = getattr(archiver, "psets", None) or {}

        # annealer metadata
        archiver.write("Annealer/beta", self.beta)
        # importance weights (set by scheduler; may be None at beta=0)
        if self.weights is not None:
            archiver.write("Annealer/weights", self.weights)

        # parameter sets
        self._record_parameter_sets(archiver=archiver, psets=psets)

        # bayesian quantities
        archiver.write("Bayesian/prior",      self.prior)
        archiver.write("Bayesian/likelihood", self.data)
        archiver.write("Bayesian/posterior",  self.posterior)

        # let subclasses record anything extra (e.g. gradients)
        self._extra_record(archiver=archiver)

        # all done
        return self

    def _record_parameter_sets(self, archiver: Archiver, psets: dict) -> None:
        """
        Write my sample matrix to the {archiver}; overridden by subclasses whose sample
        matrix needs more than a single flat dataset (e.g. reparameterization)
        """
        theta = self._shape_matrix
        if len(psets) == 0:
            archiver.write(f"ParameterSets/{self._theta_field}", theta)
        else:
            for name, pset in psets.items():
                archiver.write(f"ParameterSets/{name}",
                               theta[:, pset.offset:pset.offset+pset.count])

    def _extra_record(self, archiver: Archiver) -> None:
        """
        Hook for subclasses to record additional state (e.g. gradients); no-op by default
        """
        return

    def load_hdf5(self, path: str | None = None, iteration: int = 0) -> None:
        """
        load state from HDF5 file
        """
        # to be done
        return

# end of file
