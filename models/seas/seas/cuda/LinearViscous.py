# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-2021 parasim inc
# (c) 2010-2021 california institute of technology
# all rights reserved
#

# externals
import numpy
import pyre.cuda
# the package
import altar
import altar.cuda
# my base
from altar.models.BayesianL2 import BayesianL2
# extensions
from altar.models.seas.ext import cudaseas as libcudaseas


# declaration
class LinearViscous(BayesianL2, family="altar.models.seas.linearviscous"):
    """
    Creep model with linear viscous rheology, dv/dt = dtau/dt / alpha_1; runs on the gpu only
    """

    # configurable traits

    # model parameters
    # system information, each system consists of #patches and 2 units (slip, velocity) per patch
    patches = altar.properties.int(default=1)
    patches.doc = "number of creeping patches"
    stations = altar.properties.int(default=1)
    stations.doc = "number of surface observation locations"

    # dense output of slip rate
    t_eval_points = altar.properties.int(default=1)
    t_eval_points.doc = "number of t_eval points"
    t_eval_file = altar.properties.path(default="t_eval.txt")
    t_eval_file.doc = "the input file for time points when displacements are evaluated"

    plate_loading_velocity = altar.properties.float(default=1)
    plate_loading_velocity.doc = "Plate loading velocity, as back slip"

    stress_kernel_file = altar.properties.path(default="stresskernel.txt")
    stress_kernel_file.doc = "the filename for input stress kernel - patches x patches matrix"

    stressrate_ext_file = altar.properties.path(default="stressrate_ext.txt")
    stressrate_ext_file.doc = (r"stress rate d\tau/dt imposed by external (locked) "
                               "patches, per unit plate loading velocity - vector (patches)")

    displacement_kernel_file = altar.properties.path(default="displacementkernel.txt")
    displacement_kernel_file.doc = ("the filename for input displacement kernel G, arranged "
                                    "in (patches, stations)")

    # events - coseismic time and changes
    t_coseismic_file = altar.properties.path(default="t_coseismic.txt")
    t_coseismic_file.doc = ("the input file for the event times within a cycle, "
                            "including start/end time")
    coseismic_file = altar.properties.path(default="coseismic.txt")
    coseismic_file.doc = ("the input file for coseismic (slip, stress) changes, "
                          "matrix with (events, 2*patches) elements")

    use_spin_up_data = altar.properties.bool(default=False)
    use_spin_up_data.doc = ("whether to use a pre-computed data for (slip, velocity) "
                            "at the end of an cycle")

    spin_up_data_file = altar.properties.path(default="spin_up_data.txt")
    spin_up_data_file.doc = "the input file for spin-up data of (slip, velocity) - 2*patches"

    ode_solver_tolerance_relative = altar.properties.float(default=1e-4)
    ode_solver_tolerance_relative.doc = "max relative error for ode solver"

    ode_solver_tolerance_absolute = altar.properties.float(default=1e-3)
    ode_solver_tolerance_absolute.doc = "max absolute error for ode solver"

    spin_up_tolerance_relative = altar.properties.float(default=1e-3)
    spin_up_tolerance_relative.doc = "max relative change between cycles to stop spin up"

    spin_up_tolerance_absolute = altar.properties.float(default=1e-6)
    spin_up_tolerance_absolute.doc = "max absolute change between cycles to stop spin up"

    spin_up_max_cycles = altar.properties.int(default=5)
    spin_up_max_cycles.doc = "max number of cycles to stop spin up"

    integrator = altar.properties.str(default="dopri5")
    integrator.validators = altar.constraints.isMember("dopri5", "radau5")
    integrator.doc = "the ode integrator: dopri5 (explicit Runge-Kutta 5(4)), or radau5 " \
                     "(implicit Radau IIA, for stiff systems)"

    # protocol obligations
    @altar.export
    def initialize(self, application):
        """
        Initialize the state of the model given a {problem} specification
        """
        # the integrator is cuda only
        if altar.backends.active() != "cuda":
            application.error.log("linearviscous runs on the gpu only; set job.gpus = 1")
            raise SystemExit(1)

        # chain up
        super().initialize(application=application)

        # load files to gpu
        self.load_input_files()

        # prepare the (slip, velocity) at t_eval, in order to be accessed from python
        self.gY_eval = altar.cuda.matrix(
            shape=(self.samples, self.t_eval_points*self.patches*2), dtype=self.precision)

        # prepare the predicted data matrix
        self.gDprediction = altar.cuda.matrix(shape=(self.samples, self.observations),
                                              dtype=self.precision)

        # create the c model and pass parameters
        self.info.log(f"the model is run with {self.integrator} in precision {self.precision}")
        precision = {"float64": "double", "float32": "float"}[self.precision]
        method = "" if self.integrator == "dopri5" else f"_{self.integrator}"
        self.cmodel = getattr(libcudaseas.linearviscous, f"model_{precision}{method}")()

        # pass parameters and data to cmodel
        self.cmodel.initialize(
            self.samples, self.patches, self.stations,
            self.plate_loading_velocity,
            self.gStressKernel.grid,
            self.gStressRateExt.grid,
            self.gGF.grid,
            self.n_coseismic, self.gTCoseismic.grid, self.gCoseismic.grid,
            self.t_eval_points, self.gT_eval.grid, self.gY_eval.grid,
            self.ode_solver_tolerance_absolute,
            self.ode_solver_tolerance_relative,
            self.spin_up_tolerance_absolute,
            self.spin_up_tolerance_relative,
            self.spin_up_max_cycles
        )

        # set the initial state for spin up
        self.cmodel.set_spinup_data(self.gSpinUpData.grid)
        # all done
        return self

    def load_input_files(self):
        """
        Load All Input files
        """
        # grab the report channels
        info = self.info
        error = self.error

        # grab the sizes
        patches = self.patches
        stations = self.stations
        times = self.t_eval_points

        # load a file as a numpy array, in my precision
        def load(filename, shape):
            return numpy.array(self.io.load(filename=filename, shape=shape), dtype=self.precision)

        info.log(f'loading files from {self.case}...')
        # load the event times; there are {n_coseismic-1} events, nothing happens at the end
        t_coseismic = numpy.atleast_1d(numpy.loadtxt(self.ifs[self.t_coseismic_file].uri.path))
        self.n_coseismic = t_coseismic.size
        self.gTCoseismic = altar.cuda.vector(source=t_coseismic, dtype=self.precision)

        # load the coseismic change to (slip, stress)
        coseismic = load(self.coseismic_file, shape=((self.n_coseismic-1) * 2*patches,))
        self.gCoseismic = altar.cuda.vector(source=coseismic)

        # load the stress kernel (patches, patches)
        self.gStressKernel = altar.cuda.matrix(
            source=load(self.stress_kernel_file, shape=(patches, patches)))

        # load the stress rate imposed by external patches
        stressRateExt = load(self.stressrate_ext_file, shape=(patches,))
        # multiply by the plate loading velocity
        stressRateExt *= -self.plate_loading_velocity
        # copy to gpu
        self.gStressRateExt = altar.cuda.vector(source=stressRateExt)

        # init or load spin up data for (slips, velocities)
        if self.use_spin_up_data:
            self.gSpinUpData = altar.cuda.vector(
                source=load(self.spin_up_data_file, shape=(2*patches,)))
        else:  # set as zeros
            self.gSpinUpData = altar.cuda.vector(shape=2*patches, dtype=self.precision).zero()

        # load the displacement kernel, one copy for each t_eval point; the data covariance is
        # applied to the predictions, by {dataobs}
        GF = load(self.displacement_kernel_file, shape=(patches, stations))
        self.gGF = altar.cuda.matrix(source=numpy.tile(GF.ravel(), (times, 1)))

        # load the t_eval points
        self.gT_eval = altar.cuda.vector(source=load(self.t_eval_file, shape=(times,)))

        # check the observations add up
        if self.observations != times * stations:
            error.log(f'the number of observations {self.observations} does not match '
                      f't_eval_points x stations = {times * stations}')
            raise SystemExit(1)

        # all done
        return

    def forward_model_batched(self, theta, prediction, batch=None):
        """
        Linear Viscous forward model in batch
        :param theta: matrix (samples, parameters), physical parameters
        :param prediction: matrix (samples, observations), the predicted data
        :param batch: integer, the number of samples to be computed batch<=samples
        :return: prediction as predicted data
        """
        batch = theta.shape[0] if batch is None else batch
        parameters = theta.shape[1]
        self.cmodel.forward_model(theta.grid, prediction.grid, parameters, batch)
        # wait for it, before the host reads managed memory again
        pyre.cuda.synchronize()

        # all done
        return prediction

    def eval_data_likelihood(self, theta, likelihood, batch=None):
        """
        Compute the likelihood from my forward problem
        :param: theta - physical parameters, matrix of (samples, parameters)
        :param: likelihood - computed likelihood, vector of (samples)
        :param: batch - number of samples to be computed
        """
        # get the data storage for data prediction
        predictions = self.gDprediction

        # call forward model to calculate the data prediction
        self.forward_model_batched(theta=theta, prediction=predictions, batch=batch)

        # the l2 norm of its residual, applying the data covariance to it
        self.dataobs.eval_likelihood(prediction=predictions, likelihood=likelihood,
                                     residual=False, whitened=False, batch=batch)
        # all done
        return self

    # private data
    # inputs, on the gpu
    gT_eval = None  # the t_eval points
    gY_eval = None  # (slip, velocity) at t_eval
    gTCoseismic = None  # the event times
    gCoseismic = None  # the coseismic changes
    gStressKernel = None
    gStressRateExt = None
    gGF = None  # displacement kernel
    gSpinUpData = None
    gDprediction = None
    n_coseismic = None  # the number of event times, including start/end
    # interface to C++ model object
    cmodel = None

# end of file
