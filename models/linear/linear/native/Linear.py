# -*- python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#


# the package
import altar


# declaration
class Linear:
    """
    The cpu implementation of the linear forward model: data = G theta

    My configuration ({green}) is read directly off the shim's {model}, once, in
    {initialize}, since it's only ever needed there.
    """


    def initialize(self, model, application):
        """
        Load the Green functions and prime the residuals matrix
        """
        # load my Green functions
        self.G = model.io.load(filename=model.green, shape=(model.observations, model.parameters))
        # prepare the residuals matrix
        self.residuals = self.initialize_residuals(samples=model.samples, data=model.dataobs.dataobs)
        # all done
        return self


    def forward_model_batched(self, model, theta, prediction, batch=None):
        """
        Fill {prediction}, shape (samples x observations), with the residual G·θ - d for
        each sample in {theta}
        """
        # the green functions and the observed data
        G = self.G

        # compute G·θ^T - d, shape (observations x samples): we must transpose θ because its
        # shape is (samples x parameters) while the shape of G is (observations x parameters)
        residuals = self.residuals.clone()
        residuals = altar.blas.dgemm(G.opNoTrans, theta.opTrans, 1.0, G, theta, -1.0, residuals)

        # transpose to the (samples x observations) convention {prediction} expects
        residuals.transpose(prediction)

        # all done
        return self


    def forward_model(self, model, theta, green=None, prediction=None, observation=None):
        """
        Linear forward model prediction = G * theta for a single sample, optionally
        subtracting {observation} to get the residual instead
        """
        # resolve inputs
        green = green or self.G
        if prediction is None:
            prediction = altar.vector(shape=model.observations)

        # prediction = G * theta, optionally subtract observation
        if observation is None:
            beta = 0.0
        else:
            prediction.copy(observation)
            beta = -1.0

        altar.blas.dgemv(green.opNoTrans, 1.0, green, theta, beta, prediction)

        # all done
        return prediction


    def forward_problem(self, model, application, theta=None):
        """
        Perform the forward modeling with a given {theta}, comparing against the observed
        data; used by the {forward} action, e.g. to check the residuals of the posterior
        mean model
        """
        # load theta if not provided
        if theta is None:
            theta = model.io.load(
                filename=model.theta_input, shape=model.parameters, dataset=model.theta_dataset)

        # the residual: G*theta - d
        residual = self.forward_model(model=model, theta=theta, observation=model.dataobs.dataobs)

        # save it
        model.io.save(filename=model.forward_output, data=residual, dataset='residual')

        # all done
        return


    def gradient(self, model, controller, step, batch=None):
        """
        Fill {step.grad_prior} and {step.grad_data} with the gradients of the log prior and
        log data likelihood with respect to {step.theta}, for use by gradient-based samplers
        (e.g. SGLD)
        """
        # gradient-based samplers have no accept/reject step to catch a proposal that walked
        # outside a bounded prior's support, so they currently only support unbounded priors;
        # check once and cache, since {gradient} is called on every sweep
        if not model.checked_unbounded_priors:
            model.verify_unbounded_priors()

        # grab the portion of the sample, and of the gradient buffers, that are mine
        θ = model.restrict(theta=step.theta)
        grad_prior = model.restrict(theta=step.grad_prior)
        grad_data = model.restrict(theta=step.grad_data)

        # the prior gradient, in {psets_list} order -- {psets} is a dict and may carry extra
        # entries merged in from other configuration sources
        for name in model.psets_list:
            model.psets[name].prior_gradient(theta=θ, gradient=grad_prior)

        # the data likelihood gradient: for r = Gθ - d and Cd_inv = L (the lower Cholesky
        # factor of the inverse data covariance, so the data covariance is (L^T L)^-1),
        #     grad_data_likelihood = -G^T L^T (L r)
        G = self.G
        Cd_inv = model.dataobs.cd_inv
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
        grad_data_T = altar.matrix(shape=(model.parameters, samples)).zero()
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
    G = None # the Green functions
    residuals = None # matrix that holds (G θ - d) for each sample


# end of file
