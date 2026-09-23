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
# my base class, for its {psets}/{psets_list} and {dataobs} support
from altar.models.BayesianL2 import BayesianL2


# declaration
class Linear(BayesianL2, family="altar.models.linear"):
    """
    """


    # user configurable state
    parameters = altar.properties.int(default=None)
    parameters.doc = "the number of parameters in the model"

    # {psets_list}/{psets} and {dataobs} (observations, data covariance, norm) are inherited
    # from {BayesianL2}

    # the name of the test case
    case = altar.properties.path(default="patch-9")
    case.doc = "the directory with the input files"

    # the Green functions; the rest of the forward model's data (observations, covariance,
    # norm) is handled by {dataobs}
    green = altar.properties.path(default="green.txt")
    green.doc = "the name of the file with the Green functions"

    # settings for running the forward problem only, e.g. with the posterior mean theta
    theta_input = altar.properties.path(default="theta.txt")
    theta_input.doc = "the theta input file with a vector of parameters"

    theta_dataset = altar.properties.str(default=None)
    theta_dataset.doc = "the name/path of the theta dataset in an h5 input file"

    forward_output = altar.properties.path(default="forward_prediction.h5")
    forward_output.doc = "the name/path of the file to save forward problem results"


    # protocol obligations
    @altar.export
    def initialize(self, application):
        """
        Initialize the state of the model given a {problem} specification
        """
        # chain up; handles job/rng setup, mounting my input dataspace, {dataobs}
        # initialization (observations, data covariance, norm), and laying out my psets
        super().initialize(application=application)

        # load my Green functions
        self.G = self.io.load(filename=self.green, shape=(self.observations, self.parameters))
        # prepare the residuals matrix
        self.residuals = self.initialize_residuals(
            samples=self.samples, data=self.dataobs.dataobs)

        # grab a channel
        channel = self.debug
        channel.line("run info:")
        # show me the model
        channel.line(f" -- model: {self}")
        # the model state
        channel.line(f" -- model state:")
        channel.line(f"    parameters: {self.parameters}")
        channel.line(f"    observations: {self.observations}")
        # the test case name
        channel.line(f" -- case: {self.case}")
        # the contents of the data filesystem
        channel.line(f" -- contents of '{self.case}':")
        channel.line("\n".join(self.ifs.dump(indent=2)))
        # the loaded data
        channel.line(f" -- inputs in memory:")
        channel.line(f"    green functions: shape={self.G.shape}")
        # flush
        channel.log()

        # all done
        return self


    def forward_model_batched(self, theta, prediction):
        """
        Fill {prediction}, shape (samples x observations), with the residual G·θ - d for
        each sample in {theta}
        """
        # the green functions and the observed data
        G = self.G
        d = self.dataobs.dataobs

        # compute G·θ^T - d, shape (observations x samples): we must transpose θ because its
        # shape is (samples x parameters) while the shape of G is (observations x parameters)
        residuals = self.residuals.clone()
        residuals = altar.blas.dgemm(G.opNoTrans, theta.opTrans, 1.0, G, theta, -1.0, residuals)

        # transpose to the (samples x observations) convention {prediction} expects
        residuals.transpose(prediction)

        # all done
        return self


    def forward_model(self, theta, green=None, prediction=None, observation=None):
        """
        Linear forward model prediction = G * theta for a single sample, optionally
        subtracting {observation} to get the residual instead
        """
        # resolve inputs
        green = green or self.G
        if prediction is None:
            prediction = altar.vector(shape=self.observations)

        # prediction = G * theta, optionally subtract observation
        if observation is None:
            beta = 0.0
        else:
            prediction.copy(observation)
            beta = -1.0

        altar.blas.dgemv(green.opNoTrans, 1.0, green, theta, beta, prediction)

        # all done
        return prediction


    @altar.export
    def forward_problem(self, application, theta=None):
        """
        Perform the forward modeling with a given {theta}, comparing against the observed
        data; used by the {forward} action, e.g. to check the residuals of the posterior
        mean model
        """
        # load theta if not provided
        if theta is None:
            theta = self.io.load(
                filename=self.theta_input, shape=self.parameters, dataset=self.theta_dataset)

        # the residual: G*theta - d
        residual = self.forward_model(theta=theta, observation=self.dataobs.dataobs)

        # save it
        self.io.save(filename=self.forward_output, data=residual, dataset='residual')

        # all done
        return


    @altar.export
    def gradient(self, controller, step, batch=None):
        """
        Fill {step.grad_prior} and {step.grad_data} with the gradients of the log prior and
        log data likelihood with respect to {step.theta}, for use by gradient-based samplers
        (e.g. SGLD)
        """
        # gradient-based samplers have no accept/reject step to catch a proposal that walked
        # outside a bounded prior's support, so they currently only support unbounded priors;
        # check once and cache, since {gradient} is called on every sweep
        if not self.checked_unbounded_priors:
            self.verify_unbounded_priors()

        # grab the portion of the sample, and of the gradient buffers, that are mine
        θ = self.restrict(theta=step.theta)
        grad_prior = self.restrict(theta=step.grad_prior)
        grad_data = self.restrict(theta=step.grad_data)

        # the prior gradient, in {psets_list} order -- {psets} is a dict and may carry extra
        # entries merged in from other configuration sources
        for name in self.psets_list:
            self.psets[name].prior_gradient(theta=θ, gradient=grad_prior)

        # the data likelihood gradient: for r = Gθ - d and Cd_inv = L (the lower Cholesky
        # factor of the inverse data covariance, so the data covariance is (L^T L)^-1),
        #     grad_data_likelihood = -G^T L^T (L r)
        G = self.G
        Cd_inv = self.dataobs.cd_inv
        samples = θ.rows

        # r = Gθ^T - d, shape (observations x samples)
        r = self.residuals.clone()
        r = altar.blas.dgemm(G.opNoTrans, θ.opTrans, 1.0, G, θ, -1.0, r)
        # w = L r, then wt = L^T w (both in place)
        w = altar.blas.dtrmm(Cd_inv.sideLeft, Cd_inv.lowerTriangular, Cd_inv.opNoTrans,
                             Cd_inv.nonUnitDiagonal, 1.0, Cd_inv, r)
        wt = altar.blas.dtrmm(Cd_inv.sideLeft, Cd_inv.lowerTriangular, Cd_inv.opTrans,
                              Cd_inv.nonUnitDiagonal, 1.0, Cd_inv, w)

        # grad_T = -G^T wt, shape (parameters x samples)
        grad_data_T = altar.matrix(shape=(self.parameters, samples)).zero()
        grad_data_T = altar.blas.dgemm(G.opTrans, G.opNoTrans, -1.0, G, wt, 0.0, grad_data_T)

        # transpose to the (samples x parameters) convention {grad_data} expects
        grad_data_T.transpose(grad_data)

        # all done
        return self


    # implementation details
    def initialize_residuals(self, samples, data):
        """
        Prime the matrix that will hold the residuals (G θ - d) for each sample by duplicating the
        observation vector as many times as there are samples
        """
        # allocate the residual matrix
        r = altar.matrix(shape=(data.shape, samples))
        # for each sample
        for sample in range(samples):
            # make the corresponding column a copy of the data vector
            r.setColumn(sample, data)
        # all done
        return r


    # private data
    # inputs
    G = None # the Green functions

    # computed
    residuals = None # matrix that holds (G θ - d) for each sample


# end of file
