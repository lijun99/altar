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
# my base class
from .BayesianL2 import BayesianL2
# my protocol
from .Model import Model as model

if typing.TYPE_CHECKING:
    from altar.arrays import Array
    from altar.bayesian.controllers.Annealer import Annealer
    from altar.bayesian.states.BayesianState import BayesianState
    from altar.shells.Application import Application


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

    polar = altar.properties.list(schema=altar.properties.str(), default=[])
    polar.doc = "slips sampled as a magnitude and a rake: [magnitude, rake, along, across], my " \
                "parameter sets of the magnitudes and of the rakes, in degrees from the along " \
                "direction, and the names my models use for the two components, " \
                "along = magnitude cos(rake) and across = magnitude sin(rake)"


    # protocol obligations
    @altar.export
    def initialize(self, application: Application) -> typing.Self:
        """
        Lay out my parameter sets, then set up each of my models on its own columns
        """
        # skip {BayesianL2}'s own setup: i have no data of my own
        super(BayesianL2, self).initialize(application=application)
        self.precision = application.job.working_precision
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
        # the components of polar slips, as columns after mine
        self._width, self._polar = self.parameters, None
        if self.polar:
            if len(self.polar) != 4:
                self.error.log(f"polar: [magnitude, rake, along, across], got {self.polar}")
                raise SystemExit(1)
            magnitude, rake, along, across = self.polar
            if magnitude not in columns or rake not in columns or columns[magnitude].size != columns[rake].size:
                self.error.log(f"polar: '{magnitude}' and '{rake}' must be parameter sets of the same size")
                raise SystemExit(1)
            if along in columns or across in columns:
                self.error.log(f"polar: '{along}' and '{across}' are computed, not parameter sets")
                raise SystemExit(1)
            n = columns[magnitude].size
            columns[along] = numpy.arange(offset, offset + n)
            columns[across] = numpy.arange(offset + n, offset + 2 * n)
            self._polar = (columns[magnitude], columns[rake])
            self._width = offset + 2 * n

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
    def likelihoods(self, annealer: Annealer, step: BayesianState,
                    batch: int | None = None) -> typing.Self:
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
        step.data[...] = 0
        for name, member in self.models.items():
            likelihood = self._vector(samples=step.theta.shape[0])
            member.eval_data_likelihood(
                theta=self._member_theta(name=name, theta=step.theta, batch=samples),
                likelihood=likelihood, batch=batch)
            self._add(x=likelihood, y=step.prior if member.cascaded else step.data, batch=batch)
        dispatcher.notify(event=dispatcher.data_finish, controller=annealer)

        dispatcher.notify(event=dispatcher.posterior_start, controller=annealer)
        self.eval_posterior(step=step, batch=batch)
        dispatcher.notify(event=dispatcher.posterior_finish, controller=annealer)
        return self


    def update_model(self, annealer: Annealer, step: BayesianState) -> bool:
        """
        Let each of my models update its C_p, from its own columns of {step}
        """
        θ = self._extend(numpy.asarray(step.theta))
        updated = False
        for name, member in self.models.items():
            view = _Columns(beta=step.beta, theta=θ[:, self._columns[name]])
            updated = member.update_model(annealer=annealer, step=view) or updated
        return updated


    @altar.export
    def forward_problem(self, application: Application, theta: typing.Any) -> dict:
        """
        Each model's predictions for its columns of {theta}, keyed "<model>.<name>"
        """
        θ = self._extend(numpy.atleast_2d(numpy.asarray(theta, dtype=float)))
        out = {}
        for name, member in self.models.items():
            for key, value in member.forward_problem(
                    application=application, theta=θ[:, self._columns[name]]).items():
                out[f"{name}.{key}"] = value
        return out


    def gradient(self, controller: Annealer, step: typing.Any,
                 batch: int | None = None) -> typing.Self:
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
        if self._polar is not None:
            return self._polar_gradient(controller=controller, step=step, batch=batch)
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


    def columns(self, name: str) -> numpy.ndarray:
        """
        The columns of theta that my model {name} works on; with polar slips, of theta extended
        by their components, see {member_theta}
        """
        return self._columns[name]


    def member_theta(self, name: str, theta: numpy.ndarray) -> numpy.ndarray:
        """
        The parameters of my model {name} for the samples {theta}, (samples x parameters), on
        the host
        """
        return self._extend(numpy.atleast_2d(numpy.asarray(theta, dtype=float)))[:, self._columns[name]]


    def member_jacobian(self, name: str, theta: numpy.ndarray) -> numpy.ndarray:
        """
        The derivatives of the parameters of my model {name} with respect to mine at {theta},
        (my parameters x its parameters): its columns of the identity, but for polar slips
        """
        θ = numpy.asarray(theta, dtype=float)
        E = numpy.zeros((self.parameters, self._width))
        E[:, :self.parameters] = numpy.identity(self.parameters)
        if self._polar is not None:
            m, r = self._polar
            n = m.size
            cos, sin = numpy.cos(numpy.radians(θ[r])), numpy.sin(numpy.radians(θ[r]))
            along = self.parameters + numpy.arange(n)
            across = along + n
            E[m, along], E[r, along] = cos, -θ[m] * sin * numpy.pi / 180
            E[m, across], E[r, across] = sin, θ[m] * cos * numpy.pi / 180
        return E[:, self._columns[name]]


    # implementation details
    def _extend(self, theta: numpy.ndarray) -> numpy.ndarray:
        """
        The samples {theta}, on the host, with the components of their polar slips appended
        """
        if self._polar is None:
            return theta
        m, r = self._polar
        rake = numpy.radians(theta[:, r])
        return numpy.concatenate([theta, theta[:, m] * numpy.cos(rake), theta[:, m] * numpy.sin(rake)], axis=1)


    def _polar_gradient(self, controller: Annealer, step: typing.Any, batch: int | None = None) -> typing.Self:
        """
        The data gradients of my models with polar slips: each model's onto my parameters and
        the components, which the chain rule folds onto the magnitudes and the rakes, on the
        device; the cascaded ones, with the Jacobian of {step} if reparameterized, into its prior
        gradient
        """
        θ = step.theta
        rows = θ.shape[0]
        samples = rows if batch is None else batch
        if self._gradients is None or self._gradients[0].shape[0] != rows:
            self._gradients = (altar.cuda.matrix(shape=(rows, self._width), dtype=self.precision),
                               altar.cuda.matrix(shape=(rows, self._width), dtype=self.precision),
                               altar.cuda.matrix(shape=(rows, self.parameters), dtype=self.precision))
        data, cascade, folded = self._gradients
        data.zero()
        cascade.zero()
        for name, member in self.models.items():
            cols = self._columns[name]
            view = _Gradients(
                theta=self._member_theta(name=name, theta=θ, batch=samples),
                prior_gradient=altar.cuda.matrix(shape=(rows, cols.size), dtype=self.precision).zero(),
                data_gradient=altar.cuda.matrix(shape=(rows, cols.size), dtype=self.precision).zero())
            member.gradient(controller=controller, step=view, batch=batch)
            self._scatter(name=name, gradient=view.data_gradient, target=cascade if member.cascaded else data,
                          batch=samples)
        m, r = self._polar
        polar = altar.cuda.libcudaaltar.distributions
        polar.cudaPolar_fold(θ.grid, data.grid, step.data_gradient.grid, int(m[0]), int(r[0]), int(m.size))
        if any(member.cascaded for member in self.models.values()):
            polar.cudaPolar_fold(θ.grid, cascade.grid, folded.grid, int(m[0]), int(r[0]), int(m.size))
            jacobian = getattr(step, "Jacobian", None)
            if jacobian is not None:
                self.eval_jacobian(step=step, batch=batch)
                folded *= jacobian
            step.prior_gradient += folded
        return self


    def _member_theta(self, name: str, theta: Array, batch: int) -> Array:
        """
        The columns of {theta} my model {name} works on: {theta} itself if that is all of them,
        in order, otherwise a gather into a scratch matrix, on the device for cuda; with polar
        slips, from {theta} extended by their components, made on the device
        """
        cols = self._columns[name]
        if self._polar is None and cols.size == self.parameters and numpy.array_equal(cols, numpy.arange(self.parameters)):
            return theta
        rows = theta.shape[0]
        if altar.backends.active() != "cuda":
            return self._extend(theta)[:, cols]
        source, width = theta, self.parameters
        if self._polar is not None:
            if self._extended is None or self._extended.shape[0] != rows:
                self._extended = altar.cuda.matrix(shape=(rows, self._width), dtype=self.precision)
            m, r = self._polar
            grid = theta.grid if hasattr(theta, "grid") else theta
            altar.cuda.libcudaaltar.distributions.cudaPolar_extend(
                grid, self._extended.grid, int(m[0]), int(r[0]), int(m.size))
            source, width = self._extended, self._width
        selection, θ = self._selection(name=name, rows=rows)
        cublas = altar.cuda.cublas
        gemm = cublas.dgemm if self.precision == "float64" else cublas.sgemm
        # column-major: theta_m^T (p x n) = S^T (p x P) theta^T (P x n); the row-major S read
        # column-major is S^T already
        gemm(altar.cuda.cublas_handle(), cublas.Operation.N, cublas.Operation.N,
             cols.size, batch, width, 1.0,
             selection.grid, cols.size, source.grid, width, 0.0, θ.grid, cols.size)
        return θ


    def _scatter(self, name: str, gradient: Array, target: Array, batch: int) -> Array:
        """
        target += the gradient on the columns of my model {name}, scattered to theta's columns,
        extended by the components of polar slips
        """
        cols = self._columns[name]
        selection, _ = self._selection(name=name, rows=gradient.shape[0])
        cublas = altar.cuda.cublas
        gemm = cublas.dgemm if self.precision == "float64" else cublas.sgemm
        # column-major: target^T (W x n) += S (W x p) gradient^T (p x n), S read as its transpose
        gemm(altar.cuda.cublas_handle(), cublas.Operation.T, cublas.Operation.N,
             self._width, batch, cols.size, 1.0,
             selection.grid, cols.size, gradient.grid, cols.size, 1.0, target.grid, self._width)
        return target


    def _selection(self, name: str, rows: int) -> tuple[Array, Array]:
        """
        The (parameters x columns) selection matrix of my model {name}, theta_m = theta S, with
        theta extended by the components of polar slips, and a scratch matrix for its columns of
        theta, on the device
        """
        cols = self._columns[name]
        selection, θ = self._gather.get(name, (None, None))
        if selection is None or θ.shape[0] != rows:
            S = numpy.zeros((self._width, cols.size))
            S[cols, numpy.arange(cols.size)] = 1.0
            selection = altar.cuda.matrix(source=S, dtype=self.precision)
            θ = altar.cuda.matrix(shape=(rows, cols.size), dtype=self.precision)
            self._gather[name] = (selection, θ)
        return selection, θ


    def _vector(self, samples: int) -> Array:
        """
        A per-sample scratch vector of zeros, on my backend
        """
        if altar.backends.active() == "cuda":
            return altar.cuda.vector(shape=samples, dtype=self.precision).zero()
        return numpy.zeros(samples)


    @staticmethod
    def _add(x: Array, y: Array, batch: int | None = None) -> Array:
        """
        y += x
        """
        if altar.backends.active() == "cuda":
            altar.cuda.cublas.axpy(alpha=1.0, x=x, y=y, batch=batch)
        else:
            y += x
        return y


    # private data
    _columns: dict[str, numpy.ndarray] | None = None # the columns of theta each of my models works on
    _gather: dict | None = None # per model, the cuda selection matrix and scratch theta
    _cascade: Array | None = None # the cascaded data gradients, before their Jacobian
    _width: int = 0 # the columns of theta extended by the components of polar slips
    _polar: tuple | None = None # the columns of the magnitudes and of the rakes of polar slips
    _extended: Array | None = None # the extended theta, on the device
    _gradients: tuple | None = None # the gradients on the extended theta, and the folded one


class _Gradients:
    """
    A step as one of my models sees it, for its gradient: its columns of theta, and its own
    gradient buffers
    """

    def __init__(self, theta: Array, prior_gradient: Array, data_gradient: Array) -> None:
        self.theta = theta
        self.prior_gradient = prior_gradient
        self.data_gradient = data_gradient


class _Columns:
    """
    A step as one of my models sees it, for its C_p updates: beta, and its columns of theta
    """

    def __init__(self, beta: float, theta: numpy.ndarray) -> None:
        self.beta = beta
        self.theta = theta


# end of file
