# -*- python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
#
# (c) 2013-2020 parasim inc
# (c) 2010-2020 california institute of technology
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
    beta = None      # the inverse temperature
    theta = None     # a (samples x parameters) matrix
    prior = None     # a (samples) vector with logs of the sample likelihoods
    data = None      # a (samples) vector with the logs of the data likelihoods given the samples
    posterior = None # a (samples) vector with the logs of the posterior likelihood

    sigma = None # the parameter covariance matrix

    # the statistics of samples (theta)
    mean = None
    std = None


    # read-only public data
    @property
    def samples(self):
        """
        The number of samples
        """
        # encoded in θ
        return self.theta.rows


    @property
    def parameters(self):
        """
        The number of model parameters
        """
        # encoded in θ
        return self.theta.columns


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
        model.initializeSample(step=step)
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
    def alloc(cls, samples, parameters):
        """
        Allocate storage for the parts of a cooling step
        """
        # allocate the initial sample set
        theta = altar.matrix(shape=(samples, parameters)).zero()
        # allocate the likelihood vectors
        prior = altar.vector(shape=samples).zero()
        data = altar.vector(shape=samples).zero()
        posterior = altar.vector(shape=samples).zero()
        # build one of my instances and return it
        return cls(beta=0, theta=theta, likelihoods=(prior, data, posterior))


    # interface
    def clone(self):
        """
        Make a new step with a duplicate of my state
        """
        # make copies of my state
        beta = self.beta
        theta = self.theta.clone()
        sigma = self.sigma.clone()
        likelihoods = self.prior.clone(), self.data.clone(), self.posterior.clone()

        # make one and return it
        return type(self)(beta=beta, theta=theta, likelihoods=likelihoods, sigma=sigma)

    def computePosterior(self):
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
        θ = self.theta
        # compute the mean, sd
        self.mean, self.sd = θ.mean_sd(axis=0)
        # all done
        return self

    # meta-methods
    def __init__(self, beta, theta, likelihoods, sigma=None, **kwds):
        # chain up
        super().__init__(**kwds)

        # store the temperature
        self.beta = beta
        # store the sample set
        self.theta = theta
        # store the likelihoods
        self.prior, self.data, self.posterior = likelihoods

        # get the number of parameters
        dof = self.parameters
        # initialize the covariance matrix
        self.sigma = altar.matrix(shape=(dof,dof)).zero() if sigma is None else sigma

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
        θ = self.theta
        channel.line(f"{indent}θ: ({θ.rows} samples) x ({θ.columns} parameters)")
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

        if parameters < 10:
            # the data covariance
            Σ = self.sigma
            channel.line(f"{indent}Σ: {Σ.rows} x {Σ.columns}")
            channel.line("\n".join(Σ.print(interactive=False, indent=indent*2)))


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
        annealergrp.create_dataset('covariance', data=self.sigma.ndarray())
        # save parameter sets
        psetsgrp = f.create_group('ParameterSets')
        if len(psets) == 0 :
            # no parameter sets info provided, save as theta
            psetsgrp.create_dataset('theta', data=self.theta.ndarray())
        else:
            # get a ndarray reference for theta
            theta = self.theta.ndarray()
            # iterate over all psets
            for name, pset in psets.items():
                psetsgrp.create_dataset(name, data=theta[:, pset.offset:pset.offset+pset.count])
        # save Bayesian likelihoods/probabilities
        bayesiangrp = f.create_group('Bayesian')
        bayesiangrp.create_dataset('prior', data=self.prior.ndarray())
        bayesiangrp.create_dataset('likelihood', data=self.data.ndarray())
        bayesiangrp.create_dataset('posterior', data=self.posterior.ndarray())
        f.close()

        # all done
        return

    def load_hdf5(self, path=None, iteration=0):
        """
        load CoolingStep from HDF5 file
        """
        # to be done
        return

# end of file
