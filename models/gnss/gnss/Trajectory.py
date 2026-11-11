# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#


# externals
from __future__ import annotations
import typing
from importlib import import_module
import numpy
# the package
import altar
# my base class, for its {psets}/{psets_list} and {dataobs} support
from altar.models.BayesianL2 import BayesianL2
# the blocks and the groups of basis functions
from . import design as designs

if typing.TYPE_CHECKING:
    from altar.bayesian.controllers.Annealer import Annealer
    from altar.bayesian.states.BayesianState import BayesianState
    from altar.shells.Application import Application


# declaration
class Trajectory(BayesianL2, family="altar.models.gnss.trajectory"):
    """
    The trajectory model of many GNSS position time series ("blocks": a station and a
    component each) on a common time axis: each block is a sum of groups of basis functions,
    polynomials, sinusoids, I-splines of several scales (shared by all the blocks), steps and
    logarithmic decays (each block its own), with a coefficient per column and block

    The blocks, their observations and uncertainties, and the groups are in the case file
    {design_file}; see {altar.models.gnss.design} for its layout. Each group is a parameter set
    of the same name, of B x (its columns) parameters, block by block. The observations are
    whitened by their uncertainties in the case file, so {dataobs} reads the "data" and "mask"
    datasets of that file with cd_std = 1. The data likelihood is block diagonal; the numerics
    live in {altar.models.gnss.native.Trajectory} (cpu) or {altar.models.gnss.cuda.Trajectory}
    """


    # user configurable state
    case = altar.properties.path(default=".")
    case.doc = "the directory with the input files"

    design_file = altar.properties.path(default="trajectory.h5")
    design_file.doc = "the case file with the blocks, their observations and the groups"


    # protocol obligations
    @altar.export
    def initialize(self, application: Application) -> typing.Self:
        """
        Load my design, check my parameter sets against its groups, pick my implementation
        """
        # chain up; mounts my input dataspace, loads the observations and lays out my psets
        super().initialize(application=application)
        if self.embedded:
            self.error.log("the gnss trajectory model doesn't run inside an ensemble yet")
            raise SystemExit(1)
        # the design
        path = self.ifs[str(self.design_file)].uri.path
        self.design = designs.load(path)
        self.check(application=application)
        # where each group starts in a sample
        self.offsets = {g.name: self.psets[g.name].offset for g in self.design.groups}
        # my numerics
        backend = "cuda" if altar.backends.active() == "cuda" else "native"
        module = import_module(f"altar.models.gnss.{backend}.Trajectory")
        self._impl = getattr(module, "Trajectory")()
        self._impl.initialize(model=self, application=application)
        # report
        channel = self.info
        channel.line(f"gnss trajectory model, {backend}: '{self.design_file}'")
        for line in self.design.describe():
            channel.line(f" -- {line}")
        channel.log()
        # all done
        return self


    def check(self, application: Application) -> typing.Self:
        """
        My parameter sets must be my design's groups, with B x (columns) parameters each, and
        my observations its whitened blocks
        """
        d = self.design
        problems = []
        if self.observations != d.observations:
            problems.append(f"dataobs.observations = {self.observations}, but the case has "
                            f"{d.blocks} blocks x {d.days} days = {d.observations}")
        if self.dataobs.cd_file is not None or self.dataobs.cd_std != 1:
            problems.append("the observations are whitened in the case file: use cd_std = 1 "
                            "and no cd_file")
        groups = [g.name for g in d.groups]
        for g in d.groups:
            expected = d.blocks * g.width
            if g.name not in self.psets_list:
                problems.append(f"no parameter set '{g.name}' ({g.kind}, count = {expected})")
            elif self.psets[g.name].count != expected:
                problems.append(f"parameter set '{g.name}': count = {self.psets[g.name].count}, "
                                f"the case needs {d.blocks} blocks x {g.width} = {expected}")
        extra = [name for name in self.psets_list if name not in groups]
        if extra:
            problems.append(f"parameter sets {extra} aren't groups of the case {groups}")
        if problems:
            for p in problems:
                self.error.line(p)
            self.error.log("the configuration doesn't match the case file")
            raise SystemExit(1)
        return self


    def forward_model_batched(self, theta, prediction, batch=None):
        """
        Fill the first {batch} rows of {prediction}, (samples x observations), with the
        whitened residuals of the samples in {theta}
        """
        self._impl.forward_model_batched(model=self, theta=theta, prediction=prediction,
                                         batch=batch)
        return self


    def forward_model(self, theta, prediction):
        """
        Fill {prediction} with the whitened residuals of the single sample {theta}
        """
        prediction[:] = self.design.predict(theta, self.offsets)[0] - self.dataobs.observed()
        return self


    @altar.export
    def forward_problem(self, application, theta):
        """
        The predicted positions, (samples x observations), unwhitened, and the whitened
        residuals, for each row of {theta}; see {altar.models.Model}
        """
        θ = numpy.atleast_2d(numpy.asarray(theta, dtype=float))
        whitened = self.design.predict(θ, self.offsets)
        with numpy.errstate(divide="ignore", invalid="ignore"):
            w = self.design.weight.ravel()
            raw = numpy.where(w > 0, whitened / numpy.where(w > 0, w, 1), numpy.nan)
        # where the data are missing, predict from the unweighted design
        missing = w == 0
        if missing.any():
            unweighted = designs.Design(self.design.time, self.design.station,
                                        self.design.component,
                                        numpy.ones_like(self.design.weight), self.design.groups)
            raw[:, missing] = unweighted.predict(θ, self.offsets)[:, missing]
        return {"data": raw, "residual": whitened - self.dataobs.observed()}


    @altar.export
    def gradient(self, controller: Annealer, step: BayesianState,
                 batch: int | None = None) -> typing.Self:
        """
        Fill the prior and data gradients of {step} with respect to {step.theta}, for the
        gradient-based samplers (SGLD, HMC, MALA)
        """
        if not self.checked_unbounded_priors:
            self.verify_unbounded_priors()
        return self._impl.gradient(model=self, controller=controller, step=step, batch=batch)


    def conjugate_posterior(self, mean, variance):
        """
        Not yet: the posterior is block diagonal, a solve per block (cf_catmip)
        """
        self.error.log("the gnss trajectory model has no conjugate posterior yet; use a "
                       "gradient-based controller (langevin, hmc) or catmip")
        raise SystemExit(1)


    # private data
    design = None # the blocks and the groups, {altar.models.gnss.design.Design}
    offsets = None # where each group starts in a sample
    _impl = None # my backend implementation


# end of file
