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
import altar.cuda
from altar.cuda import libcudaaltar
# externals
import math
# my dependencies
from .BayesianState import BayesianState


# declaration
class LangevinState(BayesianState):
    """
    Encapsulation of the Langevin state of the calculation
    """

    # public data

    epsilon_t = None          # sampling rate (step size)
    prior_gradient = None     # a (samples x parameters) matrix with gradient of log P(theta_i)
    data_gradient = None      # a (samples x parameters) matrix with the gradients log likelihood log P(d|theta_i)
    eta_t = None              # a (samples x parameters) matrix with gaussian noise
    report_seq = 0

    # factories
    @classmethod
    def start(cls, controller):
        """
        Build the first cooling step by asking {model} to produce a sample set from its
        initializing prior, compute the likelihood of this sample given the data, and compute a
        (perhaps trivial) posterior
        """
        # get the model
        model = controller.model
        samples = model.job.chains
        precision = model.job.gpuprecision

        # build an uninitialized step
        step = cls.alloc(samples=samples, parameters=model.parameters, dtype=precision)

        # return the initialized state
        return step


    @classmethod
    def alloc(cls, samples, parameters, dtype):
        """
        Allocate storage for the parts of a cooling step
        """
        # dtype must be given to avoid unmatched precisions

        # allocate the initial sample set
        theta = altar.cuda.matrix(shape=(samples, parameters), dtype=dtype).zero()
        # allocate the likelihood vectors
        prior = altar.cuda.vector(shape=samples, dtype=dtype).zero()
        data = altar.cuda.vector(shape=samples, dtype=dtype).zero()
        posterior = altar.cuda.vector(shape=samples, dtype=dtype).zero()
        # allocate the langevin gradients
        prior_gradient = altar.cuda.matrix(shape=(samples, parameters), dtype=dtype).zero()
        data_gradient = altar.cuda.matrix(shape=(samples, parameters), dtype=dtype).zero()
        eta_t = altar.cuda.matrix(shape=(samples, parameters), dtype=dtype).zero()
        # build one of my instances and return it
        return cls(beta=1, theta=theta, likelihoods=(prior, data, posterior), epsilon_t=1, eta_t=eta_t, gradient=(prior_gradient, data_gradient))

    # interface
    def clone(self):
        """
        Make a new step with a duplicate of my state
        """
        # make copies of my state
        beta = self.beta
        theta = self.theta.clone()
        likelihoods = self.prior.clone(), self.data.clone(), self.posterior.clone()

        epsilon_t = self.epsilon_t
        gradient = self.prior_gradient.clone(), self.data_gradient.clone()
        eta_t = self.eta_t.clone()

        # make one and return it
        return type(self)(beta=beta, theta=theta, likelihoods=(prior, data, posterior), epsilon_t=epsilon_t, eta_t=eta_t, gradient=(prior_gradient, data_gradient))

    def computePosterior(self, batch=None):
        """
        (Re-)Compute the posterior from prior, data, and (updated) beta
        """
        batch = batch if batch is not None else self.samples
        # copy prior to posterior
        self.posterior.copy(self.prior)
        # add beta*dataLikelihood
        altar.cuda.cublas.axpy(alpha=self.beta, x=self.data, y=self.posterior, batch=batch)

        # all done
        return self

    def updateTheta(self, batch=None):
        """
        Update theta(t+1) = theta(t) + epsilon_t/2 (prior_gradient + data_gradient) + eta_t
        """
        # determine the batch size
        batch = batch or self.samples

        # generate eta_t
        epsilon_t_sqrt = math.sqrt(self.epsilon_t)
        altar.cuda.curand.gaussian(out=self.eta_t, scale=epsilon_t_sqrt)

        # theta(t+1)
        half_epsilon_t = 0.5*self.epsilon_t
        libcudaaltar.cudaLangevin_updateThetaBatched(self.theta.data,
                                    1.0, self.prior_gradient.data,
                                    1.0, self.data_gradient.data,
                                    half_epsilon_t, self.eta_t.data, batch)

        # all done
        return self

    def copyFromCPU(self, step):
        """
        Copy cpu step to gpu step
        """
        self.beta = step.beta
        self.theta.copy_from_host(source=step.theta)
        self.prior.copy_from_host(source=step.prior)
        self.data.copy_from_host(source=step.data)
        self.posterior.copy_from_host(source=step.posterior)
        return self

    def copyToCPU(self, step):
        """
        copy gpu step to cpu step
        """
        step.beta = self.beta
        self.theta.copy_to_host(target=step.theta)
        self.prior.copy_to_host(target=step.prior)
        self.data.copy_to_host(target=step.data)
        self.posterior.copy_to_host(target=step.posterior)

        return self

    def report(self, controller):
        """
        Report and Record
        """
        # report
        self.print(channel=controller.info)
        # record
        # need to compute posterior
        controller.model.likelihoods(annealer=controller, step=self)
        self.save_hdf5(path="sgld_results", iteration=self.report_seq)
        self.report_seq += 1

        # alld one
        return self



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
        channel.line(f"{indent} epsilon_t: {self.epsilon_t}")
        # the sample
        θ = self.theta
        channel.line(f"{indent}θ: ({θ.rows} samples) x ({θ.cols} parameters)")

        # print statistics (axis=0 average over samples)
        mean, sd = θ.mean_sd()
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
        annealergrp = f.create_group('Controller')
        annealergrp.create_dataset('epsilon_t', data=numpy.asarray(self.epsilon_t))
        # save parameter sets
        psetsgrp = f.create_group('ParameterSets')
        if psets is None or len(psets) == 0 :
            # no parameter sets info provided, save as theta
            psetsgrp.create_dataset('theta', data=self.theta.copy_to_host(type="numpy"))
        else:
            # get a ndarray reference for theta
            theta = self.theta.copy_to_host(type="numpy")
            # iterate over all psets
            for name, pset in psets.items():
                psetsgrp.create_dataset(name, data=theta[:, pset.offset:pset.offset+pset.count])
        # save Bayesian likelihoods/probabilities
        bayesiangrp = f.create_group('Bayesian')
        bayesiangrp.create_dataset('prior', data=self.prior.copy_to_host(type="numpy"))
        bayesiangrp.create_dataset('likelihood', data=self.data.copy_to_host(type="numpy"))
        bayesiangrp.create_dataset('posterior', data=self.posterior.copy_to_host(type="numpy"))
        f.close()

        # all done
        return

    # meta-methods
    def __init__(self, beta, theta, likelihoods, epsilon_t, eta_t, gradient, **kwds):
        # chain up
        super().__init__(**kwds)

        # store the sampling rate
        self.beta = beta
        self.epsilon_t = epsilon_t
        # store the sample set
        self.theta = theta
        # store the likelihoods
        self.prior, self.data, self.posterior = likelihoods
        # store the gaussian noise
        self.eta_t = eta_t
        self.prior_gradient, self.data_gradient = gradient

        # all done
        return

    # local
    precision = None

# end of file
