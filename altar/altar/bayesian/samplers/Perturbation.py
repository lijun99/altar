# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#


# externals
import math
from importlib import import_module
# the package
import altar
# my protocol
from .Sampler import Sampler as sampler


# declaration
class Perturbation(altar.component, family="altar.samplers.perturbation", implements=sampler):
    """
    Exact samples of a linear model with gaussian noise, under a gaussian prior on a box: its
    posterior at β is the gaussian N(m, A^-1), A = P + β G^T C^-1 G, truncated to the box.

    Each sample is a perturbation draw (Papandreou & Yuille, 2010; Orieux et al., 2012): the
    solution x of A x = P (μ + η) + β G^T C^-1 (d + ε), with η ~ N(0, P^-1) and
    ε ~ N(0, C/β), the perturbations of the prior and of the data, an independent draw of the
    posterior. Without bounds, every walk takes one such draw per chain, and the samples are
    exact and independent. With bounds, the draws are the velocities of exact hamiltonian
    trajectories, which reflect off the walls of the box (Pakman & Paninski, 2014); the chains
    then walk {stepcounter} trajectories, each one always accepted.

    The model provides its whitened linear system, {linear_system}, and its prior as a gaussian
    on a box, {gaussian_prior}: the linear and the static models, with gaussian, uniform,
    truncated gaussian or correlated priors. A, of the size of the parameters, is factored once
    per walk. At β = 1, the log evidence follows from the samples, exactly without bounds, and
    with the mass of the box estimated by GHK with them.

    Matrix-free, for models too large to factor A ({solver}): the model provides instead a
    {linear_operator} at β, an object with {size}, {apply}(x) = A x and {rhs}() = P μ + β G^T C^-1 d,
    {perturb}(rows, rng), rows of ζ + √β ε with ζ ~ N(0, P) and ε ~ N(0, G^T C^-1 G), and
    optionally {precondition}(r) ≈ A^-1 r or its own {solve}(rhs, tolerance, max_iterations),
    all on (rows x size) arrays. The mean and the draws are then solves with A, by conjugate
    gradients; the samples are exact to the {tolerance} of the solves. Unbounded priors only,
    and no log evidence.

    My implementation lives in a same-named class in {altar.bayesian.samplers.native} or
    {altar.bayesian.samplers.cuda}.
    """

    # user configurable state
    stepcounter = altar.bayesian.stepcounter()
    stepcounter.doc = "the trajectories per walk, with bounds; without, one draw is exact"

    trajectory_time = altar.properties.float(default=math.pi / 2)
    trajectory_time.validators = altar.constraints.isGreater(value=0)
    trajectory_time.doc = "the length of a trajectory, in units where a whole oscillation is " \
                          "2π; π/2, the default, lands on an independent draw without bounds"

    max_bounces = altar.properties.int(default=100000)
    max_bounces.validators = altar.constraints.isGreaterEqual(value=1)
    max_bounces.doc = "the most bounces off the walls of the box of a chain in one trajectory"

    evidence_draws = altar.properties.int(default=10000)
    evidence_draws.validators = altar.constraints.isGreaterEqual(value=0)
    evidence_draws.doc = "the draws of the GHK estimate of the mass of the box, for the log " \
                         "evidence with bounds; 0 skips the evidence with bounds"

    solver = altar.properties.str(default="auto")
    solver.validators = altar.constraints.isMember("auto", "dense", "operator")
    solver.doc = "'dense': factor A; 'operator': solve with A matrix-free, from the model's " \
                 "{linear_operator}, or from its dense system when it has none (for checks); " \
                 "'auto': the operator when the model has one"

    tolerance = altar.properties.float(default=1e-7)
    tolerance.validators = altar.constraints.isGreater(value=0)
    tolerance.doc = "matrix-free: the preconditioned residual sqrt(r^T z / r0^T z0) of the solves"

    max_iterations = altar.properties.int(default=5000)
    max_iterations.validators = altar.constraints.isGreaterEqual(value=1)
    max_iterations.doc = "matrix-free: the most conjugate-gradient iterations of a solve"


    # protocol obligations
    @altar.export
    def initialize(self, application):
        """
        Pick my backend implementation and let it initialize itself
        """
        backend = "cuda" if altar.backends.active() == "cuda" else "native"
        module = import_module(f"altar.bayesian.samplers.{backend}.Perturbation")
        self._impl = module.Perturbation()
        for trait in ("stepcounter", "trajectory_time", "max_bounces", "evidence_draws", "solver",
                      "tolerance", "max_iterations"):
            setattr(self._impl, trait, getattr(self, trait))
        self._impl.initialize(application=application)
        return self


    @altar.export
    def sample_posterior(self, annealer, step):
        """
        Draw the samples of {step} from the posterior at its β
        """
        return self._impl.sample_posterior(annealer=annealer, step=step)


    @altar.export
    def update(self, annealer, statistics):
        """
        Nothing to adapt
        """
        return self._impl.update(annealer=annealer, statistics=statistics)


    def restore(self, scaling):
        """
        Nothing to restore from an earlier run
        """
        return self


    # public data
    scaling = None # no proposal to scale

    # private data
    _impl = None # my backend implementation, chosen once, in {initialize}


# end of file
