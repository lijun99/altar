# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#


# externals
from __future__ import annotations
import json
import typing
import numpy
# the package
import altar
# my base class, for its {psets}/{psets_list} and {dataobs} support
from altar.models.BayesianL2 import BayesianL2
# the case and the numerics
from . import design as designs
from . import slip as slips

if typing.TYPE_CHECKING:
    from altar.bayesian.controllers.Annealer import Annealer
    from altar.bayesian.states.BayesianState import BayesianState
    from altar.shells.Application import Application


# declaration
class Slip(BayesianL2, family="altar.models.gnss.slip"):
    """
    Time-dependent slip on a fault mesh from the daily position time series of a GNSS network: a
    linear-gaussian model, sampled exactly, matrix-free, by the perturbation sampler

    The case file {design_file} holds the blocks (a station and a component each), their
    observations and the groups of basis functions of their trajectories (see
    {altar.models.gnss.design}). The groups named in {slip_groups}, shared by all the blocks
    (e.g. I-splines), are the temporal basis of the slip: their coefficients at block b are
    (G X)_b, G the elastic Green's functions of {greens_file}, X the slip of the N patches (dip
    slip, then strike slip) on the J basis functions. The other groups (offset, trend, seasonal,
    steps, ...) are nuisance, with gaussian priors of standard deviations {nuisance} (by group
    name, a prefix of it, or its kind), integrated out exactly per block. The noise of a block is
    white, its uncertainties in the case file, or white plus exponentially correlated processes,
    from {noise_file}. Optionally ({spatial_file}), a transient noise Xi, (B x J), correlated
    between the stations, takes part of the noise of each component.

    The prior of X is gaussian and separable: exp(-d / {length}) between the patch centroids,
    dip slip with sd sigma_j and strike slip {rho} times it, sigma_j = {sigma0}
    (s_j / {scale_ref})^{scale_power} for a column of scale s_j. The prior of Xi is matrix
    normal; see {altar.models.gnss.slip}, which holds the numerics, independent of AlTar.

    My parameter sets: "slip", 2 N J parameters, X row major; and, with {spatial_file},
    "noise", B J, Xi row major. Their {prior} components are replaced by the structured prior
    above; their {prep} draws the initial samples. The posterior has a parameter per patch,
    rake and basis function, typically 10^5 or more, so it is sampled with the {linear_operator}
    of the perturbation sampler (controller altar.bayesian.perturbation), not with a dense
    factor of its precision.
    """


    # user configurable state
    case = altar.properties.path(default=".")
    case.doc = "the directory with the input files"

    design_file = altar.properties.path(default="trajectory.h5")
    design_file.doc = "the case file with the blocks, their observations and the groups"

    greens_file = altar.properties.path(default="greens.h5")
    greens_file.doc = "the Green's functions: 'stations' (S), 'gf' (S x 3 [east, north, up] x " \
                      "2 [strike slip, dip slip] x N), 'centroids_km' (N x 3)"

    noise_file = altar.properties.path(default=None)
    noise_file.doc = "json, the correlated noise of the blocks: 'taus_days' and 'blocks' " \
                     "[{station, component, w, amps_mm}]; none: white noise, the case's sigmas"

    spatial_file = altar.properties.path(default=None)
    spatial_file.doc = "json, {component: {c0, ell_km}}: a transient noise correlated between " \
                       "the stations; none: no such noise"

    slip_groups = altar.properties.list(schema=altar.properties.str(), default=[])
    slip_groups.doc = "the groups of the case that are the temporal basis of the slip; empty: " \
                      "every I-spline group"

    nuisance = altar.properties.dict(schema=altar.properties.float())
    nuisance.doc = "the gaussian prior sd of each nuisance group, by name, name prefix or kind"

    sigma0 = altar.properties.float(default=10.0)
    sigma0.doc = "the prior sd of the dip slip of a basis function of scale {scale_ref}"

    scale_ref = altar.properties.float(default=16.0)
    scale_ref.doc = "the scale (days) at which the prior sd is {sigma0}"

    scale_power = altar.properties.float(default=0.5)
    scale_power.doc = "the prior sd grows as (scale / {scale_ref}) ** {scale_power}"

    length = altar.properties.float(default=30.0)
    length.doc = "the correlation length (km) of the slip prior between patch centroids"

    rho = altar.properties.float(default=0.3)
    rho.doc = "the prior sd of the strike slip, relative to the dip slip"

    cap = altar.properties.float(default=10.0)
    cap.doc = "the correlated noise: the sd of its directions the data hardly constrain"


    # protocol obligations
    @altar.export
    def initialize(self, application: Application) -> typing.Self:
        """
        Load the case, the Green's functions and the noise; build the posterior's numerics
        """
        super().initialize(application=application)
        if self.embedded:
            self.error.log("the gnss slip model doesn't run inside an ensemble")
            raise SystemExit(1)
        uri = lambda name: self.ifs[str(name)].uri.path
        d = self.design = designs.load(uri(self.design_file))
        groups = list(self.slip_groups) or [g.name for g in d.groups if g.kind == "isplines"]
        greens, centroids = slips.load_greens(uri(self.greens_file), d)
        noise = None
        if self.noise_file is not None and str(self.noise_file):
            taus, rows, filled = slips.load_noise(uri(self.noise_file), d)
            noise = (taus, rows)
            if filled:
                self.warning.log(f"gnss slip: {len(filled)} blocks without noise parameters "
                                 f"get their component's medians")
        spatial = None
        if self.spatial_file is not None and str(self.spatial_file):
            spatial = json.loads(open(uri(self.spatial_file)).read())
        try:
            self.numerics = slips.Slip(d, groups, dict(self.nuisance or {}), greens, centroids,
                                       noise=noise, sigma0=self.sigma0, scale_ref=self.scale_ref,
                                       power=self.scale_power, length=self.length, rho=self.rho,
                                       spatial=spatial, cap=self.cap)
        except ValueError as error:
            self.error.log(f"gnss slip: {error}")
            raise SystemExit(1)
        self.check()
        # report
        s = self.numerics
        channel = self.info
        channel.line(f"gnss slip model: '{self.design_file}', '{self.greens_file}'")
        for line in d.describe():
            channel.line(f" -- {line}")
        channel.line(f" -- slip: {s.N} patches x 2 rakes x {s.J} basis functions ({groups}), "
                     f"prior sigma0 {self.sigma0:g}, length {self.length:g} km, rho {self.rho:g}")
        channel.line(f" -- noise: {'correlated in time' if noise else 'white'}"
                     + (", correlated between stations: " + ", ".join(
                         f"{c} c0 {v['c0']:.3f}, {v['ell_km']:.0f} km" for c, v in spatial.items())
                        if spatial else ""))
        channel.line(f" -- {s.size} parameters")
        channel.log()
        return self


    def check(self) -> typing.Self:
        """
        My parameter sets: "slip", and "noise" with the correlated noise
        """
        s, d = self.numerics, self.design
        expected = {"slip": 2 * s.N * s.J}
        if s.groups:
            expected["noise"] = d.blocks * s.J
        problems = []
        if self.observations != d.observations:
            problems.append(f"dataobs.observations = {self.observations}, but the case has "
                            f"{d.blocks} blocks x {d.days} days = {d.observations}")
        if list(self.psets_list) != list(expected):
            problems.append(f"psets_list = {list(self.psets_list)}, expected {list(expected)}")
        for name, count in expected.items():
            if name in self.psets and self.psets[name].count != count:
                problems.append(f"parameter set '{name}': count = {self.psets[name].count}, "
                                f"expected {count}")
            if name in self.psets and self.psets[name].offset != (0 if name == "slip"
                                                                   else expected["slip"]):
                problems.append(f"parameter set '{name}' is out of place")
        if problems:
            for p in problems:
                self.error.line(p)
            self.error.log("the configuration doesn't match the case")
            raise SystemExit(1)
        return self


    # the densities
    @altar.export
    def eval_prior(self, step: BayesianState, batch: int | None = None) -> typing.Self:
        """
        Add the log prior of the samples of {step} to {step.prior}
        """
        θ = self._host(self.restrict(theta=step.theta), batch)
        self._add(step.prior, self.numerics.log_prior(θ), batch)
        return self


    def eval_prior_physical(self, step: BayesianState, batch: int | None = None) -> typing.Self:
        """
        No reparameterization: the same as {eval_prior}
        """
        return self.eval_prior(step=step, batch=batch)


    def eval_data_likelihood(self, theta, likelihood, batch=None) -> typing.Self:
        """
        Fill {likelihood} with the log likelihood of the data, the nuisance integrated out
        """
        θ = self._host(theta, batch)
        self._set(likelihood, self.numerics.log_likelihood(θ), batch)
        return self


    # the samplers of linear-gaussian posteriors
    def linear_operator(self, beta: float = 1.0) -> typing.Any:
        """
        My posterior precision at {beta}, matrix-free; see {altar.models.gnss.slip.Operator}
        """
        return self.numerics.operator(beta)


    def gaussian_prior(self):
        raise NotImplementedError("the gnss slip model is too large for a dense prior; use the "
                                  "matrix-free solver of the perturbation sampler")


    def linear_system(self):
        raise NotImplementedError("the gnss slip model is too large for a dense system; use the "
                                  "matrix-free solver of the perturbation sampler")


    @altar.export
    def forward_problem(self, application, theta):
        """
        The displacement of the slip groups at every block and day (samples x observations, the
        case's units), and of the slip alone
        """
        θ = numpy.atleast_2d(numpy.asarray(theta, dtype=float))
        s, d = self.numerics, self.design
        basis = numpy.hstack([g.G for name in s.slip_groups for g in d.groups if g.name == name])
        X, _ = s.split(θ)
        slip_only = numpy.einsum("kbj,nj->kbn", numpy.matmul(s.G, X), basis)
        total = numpy.einsum("kbj,nj->kbn", s.coefficients(θ), basis)
        k = θ.shape[0]
        return {"data": total.reshape(k, -1), "slip": slip_only.reshape(k, -1)}


    # implementation details
    def _host(self, theta, batch):
        rows = theta.shape[0] if batch is None else batch
        if isinstance(theta, numpy.ndarray):
            return theta[:rows]
        return numpy.asarray(theta.copy_to_host() if hasattr(theta, "copy_to_host")
                             else theta)[:rows]


    def _add(self, out, values, batch):
        if isinstance(out, numpy.ndarray):
            out[:values.size] += values
        else:
            raise NotImplementedError("the gnss slip model's densities run on the host")


    def _set(self, out, values, batch):
        if isinstance(out, numpy.ndarray):
            out[:values.size] = values
        else:
            raise NotImplementedError("the gnss slip model's densities run on the host")


    # private data
    design = None # the case, {altar.models.gnss.design.Design}
    numerics = None # the posterior, {altar.models.gnss.slip.Slip}


# end of file
