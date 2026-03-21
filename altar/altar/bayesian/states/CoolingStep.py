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


# declaration
class CoolingStep:
    """
    Encapsulation of the state of the calculation at some particular β value
    """


    # public data
    beta = None      # the inverse temperature, from tempering schedule 
    theta = None  # a (samples x parameters) matrix in physical space   (theta)
    theta_sampling = None     # a (samples x parameters) matrix in sampling space (phi)
    prior = None     # a (samples) vector with the log of the prior probabilities P(theta) 
    jacobian = None  # a (samples) vector with the log of the Jacobian determinant (d theta/d phi)    
    data = None      # a (samples) vector with the logs of the data likelihoods given the samples 
    posterior = None # a (samples) vector with the logs of the posterior likelihood 
                     # P(phi|data) during simulation, tempered by beta
                     
    # reparameterization flag
    has_reparametrization = False  # whether reparameterization is implemented

    weights = None  # a (samples) vector of importance weights w_i ∝ exp(Δβ · data_i); set by scheduler

    # the statistics of samples (theta)
    mean = None
    sd = None


    # read-only public data
    @property
    def samples(self):
        """
        The number of samples
        """
        # encoded in θ_sampling
        return self.theta_sampling.rows


    @property
    def parameters(self):
        """
        The number of model parameters
        """
        # encoded in θ_sampling
        return self.theta_sampling.columns


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
        prior = altar.vector(shape=samples).zero()
        data = altar.vector(shape=samples).zero()
        posterior = altar.vector(shape=samples).zero()

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
        # get the samples in sampling space
        θ = self.theta_sampling
        # compute the mean, sd
        self.mean, self.sd = θ.mean_sd(axis=0)
        # all done
        return self

    # meta-methods
    def __init__(self, beta, theta_sampling=None, theta=None, jacobian=None, likelihoods=None, has_reparametrization=False, **kwds):
        # chain up
        super().__init__(**kwds)

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
        # the sample in sampling space
        θ = self.theta_sampling
        channel.line(f"{indent}θ_sampling: ({θ.rows} samples) x ({θ.columns} parameters)")
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
        # flush
        channel.log()

        # all done
        return channel

    def save_hdf5(self, path=None, iteration=None, psets=None):
        """
        Save Coolinging Step to HDF5 file
        Args:
            step altar.bayesian.CoolingStep
            path altar.primitives.path
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
        # save Bayesian likelihoods/probabilities
        bayesiangrp = f.create_group('Bayesian')
        bayesiangrp.create_dataset('prior', data=self.prior.ndarray())
        bayesiangrp.create_dataset('likelihood', data=self.data.ndarray())
        bayesiangrp.create_dataset('posterior', data=self.posterior.ndarray())
        f.close()

        # all done
        return

    def record(self, archiver):
        """
        Record me using the provided {archiver}.
        """
        import numpy

        psets = getattr(archiver, "psets", None) or {}

        # annealer metadata
        archiver.write("Annealer/beta", self.beta)
        archiver.write("ParameterSets/has_reparametrization",
                       numpy.array([self.has_reparametrization]))

        # importance weights (set by scheduler; may be None at beta=0)
        if self.weights is not None:
            archiver.write("Annealer/weights", self.weights)

        # parameter sets
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

        # bayesian quantities
        archiver.write("Bayesian/prior",      self.prior)
        archiver.write("Bayesian/likelihood", self.data)
        archiver.write("Bayesian/posterior",  self.posterior)

        # all done
        return self

    def load_hdf5(self, path=None, iteration=0):
        """
        load CoolingStep from HDF5 file
        """
        # to be done
        return

# end of file
