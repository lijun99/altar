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
import numpy
# the package
import altar
# my base class
from .BayesianL2 import BayesianL2
# my protocol
from .Model import Model as model


# declaration
class Ensemble(BayesianL2, family="altar.models.ensemble"):
    """
    An ensemble of models sharing one set of parameters, e.g. a cascaded {static, kinematic}
    slip inversion

    I own the parameter sets; each of my {models} names the ones it uses in its own
    {psets_list}, in the order it expects them, and computes its data likelihood on just those
    columns of theta. A {cascaded} model contributes at beta = 1, as part of the prior; the
    others are annealed.
    """


    # user configurable state
    models = altar.properties.dict(schema=model())
    models.doc = "the models in this ensemble, each with its own data"


    # protocol obligations
    @altar.export
    def initialize(self, application):
        """
        Lay out my parameter sets, then set up each of my models on its own columns
        """
        # skip {BayesianL2}'s own setup: i have no data of my own
        super(BayesianL2, self).initialize(application=application)
        self.precision = application.job.gpuprecision
        self.ifs = self.mount_input_dataspace(pfs=application.pfs)
        self.io = altar.io.FileIO(ifs=self.ifs, error=self.error, precision=self.precision)
        self.samples = application.job.chains
        self.parameters = self.initialize_psets(application=application)

        # the columns of each of my parameter sets
        self._gather = {}
        columns, offset = {}, 0
        for name in self.psets_list:
            count = self.psets[name].count
            columns[name] = numpy.arange(offset, offset + count)
            offset += count

        # my models, each on the columns of the parameter sets it names
        self._columns = {}
        for name, member in self.models.items():
            missing = [pset for pset in member.psets_list or [] if pset not in columns]
            if not member.psets_list or missing:
                self.error.log(f"ensemble model '{name}' must name its parameter sets, from "
                               f"{self.psets_list}, in its psets_list; unknown: {missing}")
                raise SystemExit(1)
            cols = numpy.concatenate([columns[pset] for pset in member.psets_list])
            self._columns[name] = cols
            member.embedded = True
            member.parameters = int(cols.size)
            member.initialize(application=application)
        # the members mounted their own inputs; mine are the ones that stay visible
        application.pfs["inputs"] = self.ifs.discover()
        return self


    @altar.export
    def likelihoods(self, annealer, step, batch=None):
        """
        The prior from my parameter sets, plus each cascaded model's data likelihood; the data
        likelihood of the others; and the posterior at {step.beta}
        """
        dispatcher = annealer.dispatcher
        samples = step.theta.shape[0] if batch is None else batch

        dispatcher.notify(event=dispatcher.prior_start, controller=annealer)
        self.eval_prior(step=step, batch=batch)
        dispatcher.notify(event=dispatcher.prior_finish, controller=annealer)

        dispatcher.notify(event=dispatcher.data_start, controller=annealer)
        step.data.zero()
        for name, member in self.models.items():
            likelihood = self._vector(samples=step.theta.shape[0]).zero()
            member.eval_data_likelihood(
                theta=self._member_theta(name=name, theta=step.theta, batch=samples),
                likelihood=likelihood, batch=batch)
            self._add(x=likelihood, y=step.prior if member.cascaded else step.data, batch=batch)
        dispatcher.notify(event=dispatcher.data_finish, controller=annealer)

        dispatcher.notify(event=dispatcher.posterior_start, controller=annealer)
        self.eval_posterior(step=step, batch=batch)
        dispatcher.notify(event=dispatcher.posterior_finish, controller=annealer)
        return self


    def update_model(self, annealer, step):
        """
        Let each of my models update its C_p, from its own columns of {step}
        """
        θ = numpy.asarray(step.theta)
        updated = False
        for name, member in self.models.items():
            view = _Columns(beta=step.beta, theta=θ[:, self._columns[name]])
            updated = member.update_model(annealer=annealer, step=view) or updated
        return updated


    @altar.export
    def forward_problem(self, application, theta):
        """
        Each model's predictions for its columns of {theta}, keyed "<model>.<name>"
        """
        θ = numpy.atleast_2d(numpy.asarray(theta, dtype=float))
        out = {}
        for name, member in self.models.items():
            for key, value in member.forward_problem(
                    application=application, theta=θ[:, self._columns[name]]).items():
                out[f"{name}.{key}"] = value
        return out


    def gradient(self, controller, step, batch=None):
        """
        The prior gradient from my parameter sets, plus the data gradient of each cascaded model;
        the data gradient of the others; each model's on its own columns, scattered back to
        theta's (gpu only). The samplers chain only {data_gradient} into sampling space, so when
        {step} is reparameterized the cascaded data gradients get its Jacobian here
        """
        if altar.backends.active() != "cuda":
            raise NotImplementedError("ensemble gradients need the gpu (job.gpus = 1)")
        if not self.checked_unbounded_priors:
            self.verify_unbounded_priors()
        θ = step.theta
        rows = θ.shape[0]
        samples = rows if batch is None else batch
        for name in self.psets_list:
            self.psets[name].prior_gradient(theta=θ, gradient=step.prior_gradient, batch=batch)
        step.data_gradient.zero()
        # the cascaded data gradients, gathered apart when they need the Jacobian
        jacobian = getattr(step, "Jacobian", None)
        cascade = None
        if jacobian is not None and any(member.cascaded for member in self.models.values()):
            self.eval_jacobian(step=step, batch=batch)
            if self._cascade is None or self._cascade.shape != θ.shape:
                self._cascade = altar.cuda.matrix(shape=θ.shape, dtype=self.precision)
            cascade = self._cascade.zero()
        for name, member in self.models.items():
            cols = self._columns[name]
            view = _Gradients(
                theta=self._member_theta(name=name, theta=θ, batch=samples),
                prior_gradient=altar.cuda.matrix(shape=(rows, cols.size), dtype=self.precision).zero(),
                data_gradient=altar.cuda.matrix(shape=(rows, cols.size), dtype=self.precision).zero())
            member.gradient(controller=controller, step=view, batch=batch)
            target = step.data_gradient if not member.cascaded else (
                cascade if cascade is not None else step.prior_gradient)
            self._scatter(name=name, gradient=view.data_gradient, target=target, batch=samples)
        # d(physical)/d(sampling) on the cascaded data gradients, into the prior gradient
        if cascade is not None:
            cascade *= jacobian
            step.prior_gradient += cascade
        return self


    def columns(self, name):
        """
        The columns of theta that my model {name} works on
        """
        return self._columns[name]


    # implementation details
    def _member_theta(self, name, theta, batch):
        """
        The columns of {theta} my model {name} works on: {theta} itself if that is all of them,
        in order, otherwise a gather into a scratch matrix, on the device for cuda
        """
        cols = self._columns[name]
        if cols.size == self.parameters and numpy.array_equal(cols, numpy.arange(self.parameters)):
            return theta
        rows = theta.shape[0]
        if altar.backends.active() != "cuda":
            θ = altar.matrix(shape=(rows, cols.size))
            numpy.asarray(θ)[:, :] = numpy.asarray(theta)[:, cols]
            return θ
        selection, θ = self._selection(name=name, rows=rows)
        cublas = altar.cuda.cublas
        gemm = cublas.dgemm if self.precision == "float64" else cublas.sgemm
        # column-major: theta_m^T (p x n) = S^T (p x P) theta^T (P x n); the row-major S read
        # column-major is S^T already
        gemm(altar.cuda.cublas_handle(), cublas.Operation.N, cublas.Operation.N,
             cols.size, batch, self.parameters, 1.0,
             selection.grid, cols.size, theta.grid, self.parameters, 0.0, θ.grid, cols.size)
        return θ


    def _scatter(self, name, gradient, target, batch):
        """
        target += the gradient on the columns of my model {name}, scattered to theta's columns
        """
        cols = self._columns[name]
        selection, _ = self._selection(name=name, rows=gradient.shape[0])
        cublas = altar.cuda.cublas
        gemm = cublas.dgemm if self.precision == "float64" else cublas.sgemm
        # column-major: target^T (P x n) += S (P x p) gradient^T (p x n), S read as its transpose
        gemm(altar.cuda.cublas_handle(), cublas.Operation.T, cublas.Operation.N,
             self.parameters, batch, cols.size, 1.0,
             selection.grid, cols.size, gradient.grid, cols.size, 1.0, target.grid, self.parameters)
        return target


    def _selection(self, name, rows):
        """
        The (parameters x columns) selection matrix of my model {name}, theta_m = theta S, and a
        scratch matrix for its columns of theta, on the device
        """
        cols = self._columns[name]
        selection, θ = self._gather.get(name, (None, None))
        if selection is None or θ.shape[0] != rows:
            S = numpy.zeros((self.parameters, cols.size))
            S[cols, numpy.arange(cols.size)] = 1.0
            selection = altar.cuda.matrix(source=S, dtype=self.precision)
            θ = altar.cuda.matrix(shape=(rows, cols.size), dtype=self.precision)
            self._gather[name] = (selection, θ)
        return selection, θ


    def _vector(self, samples):
        """
        A per-sample scratch vector, on my backend
        """
        if altar.backends.active() == "cuda":
            return altar.cuda.vector(shape=samples, dtype=self.precision)
        return altar.vector(shape=samples)


    @staticmethod
    def _add(x, y, batch=None):
        """
        y += x
        """
        if altar.backends.active() == "cuda":
            altar.cuda.cublas.axpy(alpha=1.0, x=x, y=y, batch=batch)
        else:
            altar.blas.daxpy(1.0, x, y)
        return y


    # private data
    _columns = None # the columns of theta each of my models works on
    _gather = None # per model, the cuda selection matrix and scratch theta
    _cascade = None # the cascaded data gradients, before their Jacobian


class _Gradients:
    """
    A step as one of my models sees it, for its gradient: its columns of theta, and its own
    gradient buffers
    """

    def __init__(self, theta, prior_gradient, data_gradient):
        self.theta = theta
        self.prior_gradient = prior_gradient
        self.data_gradient = data_gradient


class _Columns:
    """
    A step as one of my models sees it, for its C_p updates: beta, and its columns of theta
    """

    def __init__(self, beta, theta):
        self.beta = beta
        self.theta = theta


# end of file
