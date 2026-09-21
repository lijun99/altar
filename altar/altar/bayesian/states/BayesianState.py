# -*- python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# lijun zhu <ljzhu@caltech.edu>
#
# (c) 2013-2021 parasim inc
# (c) 2010-2021 california institute of technology
# all rights reserved
#


# the package
import altar


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
    theta = None     # a (samples x parameters) matrix
    prior = None     # a (samples) vector with logs of the prior
    data = None      # a (samples) vector with the logs of the data likelihoods given the samples
    posterior = None # a (samples) vector with the logs of the posterior
    weights = None   # a (samples) vector of importance weights w_i ∝ exp(Δβ · data_i); set by scheduler

    # the statistics of samples
    mean = None
    sd = None

    # the name of the attribute that holds the (samples x parameters) sample matrix, and
    # the label used for it in diagnostic output; subclasses whose sample matrix lives
    # under a different name (e.g. {CoolingStep.theta_sampling}) override these
    _theta_field = "theta"
    _theta_label = "θ"


    # read-only public data
    @property
    def _shape_matrix(self):
        """
        The (samples x parameters) matrix that determines my shape
        """
        return getattr(self, self._theta_field)

    @property
    def samples(self):
        """
        The number of samples
        """
        return self._shape_matrix.rows

    @property
    def parameters(self):
        """
        The number of model parameters
        """
        return self._shape_matrix.columns


    @classmethod
    def start(cls, annealer):
        """
        Build the first cooling step by asking {model} to produce a sample set from its
        initializing prior, compute the likelihood of this sample given the data, and compute a
        (perhaps trivial) posterior
        """
        # get the model
        model = annealer.model
        # build an uninitialized step
        step = cls.alloc(samples=model.job.chains, parameters=model.parameters)

        # initialize it
        model.initialize_sample(step=step)
        # compute the likelihoods
        model.likelihoods(annealer=annealer, step=step)
        # let subclasses do any extra work that depends on the likelihoods being ready
        step._on_start(annealer=annealer)

        step.prior.print()

        # return the initialized state
        return step

    @classmethod
    def allocate(cls, annealer):
        # get the model
        model = annealer.model
        # build an uninitialized step
        step = cls.alloc(samples=model.job.chains, parameters=model.parameters)
        return step

    @classmethod
    def alloc(cls, samples, parameters):
        """
        Allocate storage for the parts of a cooling step
        """
        # allocate the initial sample set
        theta = altar.matrix(shape=(samples, parameters)).zero()
        # allocate the likelihood vectors
        prior, data, posterior = cls._alloc_likelihoods(samples)
        # build one of my instances and return it
        return cls(beta=0, theta=theta, likelihoods=(prior, data, posterior))

    @classmethod
    def _alloc_likelihoods(cls, samples):
        """
        Allocate the (samples) prior/data/posterior likelihood vectors
        """
        prior = altar.vector(shape=samples).zero()
        data = altar.vector(shape=samples).zero()
        posterior = altar.vector(shape=samples).zero()
        return prior, data, posterior

    def _on_start(self, annealer):
        """
        Hook invoked by {start} right after the likelihoods have been computed; the base
        implementation does nothing. {HMCState} uses this to compute gradients.
        """
        return


    # interface
    def clone(self):
        """
        Make a new step with a duplicate of my state
        """
        # make copies of my state
        beta = self.beta
        theta = self.theta.clone()
        likelihoods = self.prior.clone(), self.data.clone(), self.posterior.clone()

        # make one and return it
        return type(self)(beta=beta, theta=theta, likelihoods=likelihoods)

    def compute_posterior(self):
        """
        Compute the posterior from prior, data, and beta
        """

        # in their log form, posterior = prior + beta * datalikelihood
        # make a copy of prior at first
        self.posterior.copy(self.prior)
        # add the data likelihood
        altar.blas.daxpy(self.beta, self.data, self.posterior)
        # all done
        return self

    def statistics(self):
        """
        Compute the statistics of samples
        :return:
        """
        # get the samples
        θ = self._shape_matrix
        # compute the mean, sd
        self.mean, self.sd = θ.mean_sd(axis=0)
        # all done
        return self

    # meta-methods
    def __init__(self, beta, theta, likelihoods, **kwds):
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
    def print(self, channel, indent=' '*2):
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
        channel.line(f"{indent}{self._theta_label}: ({θ.rows} samples) x ({θ.columns} parameters)")
        if θ.rows <= 10 and θ.columns <= 10:
            channel.line("\n".join(θ.print(interactive=False, indent=indent*2)))

        if samples < 10:
            # the prior
            prior = self.prior
            channel.line(f"{indent}prior:")
            channel.line(prior.print(interactive=False, indent=indent*2))
            # the data
            data = self.data
            channel.line(f"{indent}data:")
            channel.line(data.print(interactive=False, indent=indent*2))
            # the posterior
            posterior = self.posterior
            channel.line(f"{indent}posterior:")
            channel.line(posterior.print(interactive=False, indent=indent*2))

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

    def _extra_print(self, channel, indent):
        """
        Hook for subclasses to print additional state; no-op by default
        """
        return

    def save_hdf5(self, path=None, iteration=None, psets=None):
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
        import numpy

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
        bayesiangrp.create_dataset('prior', data=self.prior.ndarray())
        bayesiangrp.create_dataset('likelihood', data=self.data.ndarray())
        bayesiangrp.create_dataset('posterior', data=self.posterior.ndarray())
        # let subclasses save anything extra (e.g. gradients)
        self._extra_save_hdf5(f=f)
        f.close()

        # all done
        return

    def _save_parameter_sets_hdf5(self, psetsgrp, psets):
        """
        Write my sample matrix into the "ParameterSets" hdf5 group; overridden by subclasses
        whose sample matrix needs more than a single flat dataset (e.g. reparameterization)
        """
        theta = self._shape_matrix
        if len(psets) == 0:
            psetsgrp.create_dataset(self._theta_field, data=theta.ndarray())
        else:
            theta_arr = theta.ndarray()
            for name, pset in psets.items():
                psetsgrp.create_dataset(name, data=theta_arr[:, pset.offset:pset.offset+pset.count])

    def _extra_save_hdf5(self, f):
        """
        Hook for subclasses to persist additional state (e.g. gradients); no-op by default
        """
        return

    def record(self, archiver):
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

    def _record_parameter_sets(self, archiver, psets):
        """
        Write my sample matrix to the {archiver}; overridden by subclasses whose sample
        matrix needs more than a single flat dataset (e.g. reparameterization)
        """
        theta = self._shape_matrix
        if len(psets) == 0:
            archiver.write(f"ParameterSets/{self._theta_field}", theta)
        else:
            theta_arr = theta.ndarray()
            for name, pset in psets.items():
                archiver.write(f"ParameterSets/{name}",
                               theta_arr[:, pset.offset:pset.offset+pset.count])

    def _extra_record(self, archiver):
        """
        Hook for subclasses to record additional state (e.g. gradients); no-op by default
        """
        return

    def load_hdf5(self, path=None, iteration=0):
        """
        load state from HDF5 file
        """
        # to be done
        return

# end of file
