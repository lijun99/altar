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

# declaration
class cudaCoolingStep:
    """
    Encapsulation of the state of the calculation at some particular β value
    """

    # public data
    beta = None      # the inverse temperature
    theta_sampling = None     # a (samples x parameters) matrix in sampling space
    theta = None  # a (samples x parameters) matrix in physical space
    jacobian = None  # a (samples) vector with the log of the Jacobian determinant
    prior = None     # a (samples) vector with logs of the sample likelihoods
    data = None      # a (samples) vector with the logs of the data likelihoods given the samples
    posterior = None # a (samples) vector with the logs of the posterior likelihood

    # reparameterization flag
    has_reparametrization = False  # whether reparameterization is implemented

    # read-only public data
    @property
    def samples(self):
        """
        The number of samples
        """
        # encoded in θ_sampling
        return self.theta_sampling.shape[0]


    @property
    def parameters(self):
        """
        The number of model parameters
        """
        # encoded in θ_sampling
        return self.theta_sampling.shape[1]


    # factories
    @classmethod
    def start(cls, annealer):
        """
        Build the first cooling step by asking {model} to produce a sample set from its
        initializing prior, compute the likelihood of this sample given the data, and compute a
        (perhaps trivial) posterior
        """
        # get the model
        model = annealer.model
        samples = model.job.chains
        precision = model.job.gpuprecision

        # build an uninitialized step
        step = cls.alloc(samples=samples, parameters=model.parameters, dtype=precision,
                        has_reparametrization=model.has_reparametrization if hasattr(model, 'has_reparametrization') else False)

        # return the initialized state
        return step


    @classmethod
    def alloc(cls, samples, parameters, dtype, has_reparametrization=False):
        """
        Allocate storage for the parts of a cooling step
        """
        # dtype must be given to avoid unmatched precisions

        # allocate the initial sample set in sampling space
        theta_sampling = altar.cuda.matrix(shape=(samples, parameters), dtype=dtype).zero()

        # allocate physical parameters and jacobian only if using reparameterization
        theta = None
        jacobian = None
        if has_reparametrization:
            theta = altar.cuda.matrix(shape=(samples, parameters), dtype=dtype).zero()
            jacobian = altar.cuda.vector(shape=samples, dtype=dtype).zero()

        # allocate the likelihood vectors
        prior = altar.cuda.vector(shape=samples, dtype=dtype).zero()
        data = altar.cuda.vector(shape=samples, dtype=dtype).zero()
        posterior = altar.cuda.vector(shape=samples, dtype=dtype).zero()

        # build one of my instances and return it
        return cls(beta=0, theta_sampling=theta_sampling, theta=theta, jacobian=jacobian,
                  likelihoods=(prior, data, posterior), has_reparametrization=has_reparametrization)


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

    def copyFromCPU(self, step):
        """
        Copy cpu step to gpu step
        """
        self.beta = step.beta
        self.theta_sampling.copy_from_host(source=step.theta_sampling)
        if self.has_reparametrization:
            self.theta.copy_from_host(source=step.theta)
            self.jacobian.copy_from_host(source=step.jacobian)
        self.prior.copy_from_host(source=step.prior)
        self.data.copy_from_host(source=step.data)
        self.posterior.copy_from_host(source=step.posterior)
        return self

    def copyToCPU(self, step):
        """
        copy gpu step to cpu step
        """
        step.beta = self.beta
        self.theta_sampling.copy_to_host(target=step.theta_sampling)
        if self.has_reparametrization:
            self.theta.copy_to_host(target=step.theta)
            self.jacobian.copy_to_host(target=step.jacobian)
        self.prior.copy_to_host(target=step.prior)
        self.data.copy_to_host(target=step.data)
        self.posterior.copy_to_host(target=step.posterior)

        return self

    # meta-methods
    def __init__(self, beta, theta_sampling=None, theta=None, jacobian=None, likelihoods=None, has_reparametrization=False, **kwds):
        # chain up
        super().__init__(**kwds)

        # store the temperature
        self.beta = beta
        # store the sample sets
        self.theta_sampling = theta_sampling
        # store reparameterization flag
        self.has_reparametrization = has_reparametrization

        # handle physical parameters and jacobian based on reparameterization flag
        if has_reparametrization:
            self.theta = theta if theta is not None else theta_sampling.clone()
            self.jacobian = jacobian if jacobian is not None else altar.cuda.vector(shape=theta_sampling.shape[0], dtype=theta_sampling.dtype).zero()
        else:
            # if no reparameterization, physical parameters are the same as sampling parameters
            self.theta = self.theta_sampling
            self.jacobian = None

        # store the likelihoods
        self.prior, self.data, self.posterior = likelihoods

        # all done
        return

    # local
    precision = None

# end of file
