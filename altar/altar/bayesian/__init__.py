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
# protocols (re-exported for use as altar.bayesian.*)
from .controllers.Controller import Controller as controller
from .samplers.Sampler import Sampler as sampler
from .schedulers.Scheduler import Scheduler as scheduler
from .proposals.Proposal import Proposal as proposal
# from altar.simulations.Monitor import Monitor as monitor
# from altar.simulations.Archiver import Archiver as archiver
from .solvers.Solver import Solver as solver
from .langevin_schedulers.LangevinScheduler import LangevinScheduler as langevinscheduler
from .stepsizers.StepSizer import StepSizer as stepsizer
from .stepcounters.StepCounter import StepCounter as stepcounter

# foundry registrations — re-export from each subpackage into altar.bayesian namespace
from .controllers import annealer, catmip, langevin, mcmc, plainhmc, catmiphmc
from .samplers import metropolis, hmc
from .schedulers import constanttemperature, cov
from .solvers import brent, grid
from .proposals import gaussianproposal
from .stepsizers import fixedstep, linearrate, targetedrate, dual
from .stepcounters import fixedsteps, decorrelating
from .monitoring import profiler
from .archivers import recorder, h5recorder
from .langevin_schedulers import powerdecay, expdecay

# end of file
