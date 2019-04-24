# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-2019 parasim inc
# (c) 2010-2019 california institute of technology
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
    theta = None     # a (samples x parameters) matrix
    prior = None     # a (samples) vector with logs of the sample likelihoods
    data = None      # a (samples) vector with the logs of the data likelihoods given the samples
    posterior = None # a (samples) vector with the logs of the posterior likelihood

    # read-only public data
    @property
    def samples(self):
        """
        The number of samples
        """
        # encoded in θ
        return self.theta.shape[0]


    @property
    def parameters(self):
        """
        The number of model parameters
        """
        # encoded in θ
        return self.theta.shape[1]


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
        step = cls.alloc(samples=samples, parameters=model.parameters, dtype=precision)

        # initialize it
        model.cuInitSample(theta=step.theta)
        # compute the likelihoods
        model.likelihoods(annealer=annealer, step=step, batch=samples)

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
        likelihoods = self.prior.clone(), self.data.clone(), self.posterior.clone()

        # make one and return it
        return type(self)(beta=beta, theta=theta, likelihoods=likelihoods)

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

    # local
    precision = None

# end of file
