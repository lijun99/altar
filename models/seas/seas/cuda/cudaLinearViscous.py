# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-2021 parasim inc
# (c) 2010-2021 california institute of technology
# all rights reserved
#

# the package
import altar
import altar.cuda
# my base
from altar.cuda.models.cudaBayesian import cudaBayesian
# extensions
# from altar.cuda import cublas
# from altar.cuda import libcuda
from altar.models.seas.ext import cudaseas as libcudaseas
import numpy


# declaration
class cudaLinearViscous(cudaBayesian, family="altar.models.seas.cuda.linearviscous"):
    """
    Creep model with linear viscous rheology
    """

    # configurable traits

    # data observations
    # flattened 1d vector with time points x stations
    dataobs = altar.cuda.data.data()
    dataobs.default = altar.cuda.data.datal2()
    dataobs.doc = "the observed data"

    # model parameters
    # system information, each system consists of #patches and 2 units (slip, velocity) per patch
    patches = altar.properties.int(default=1)
    patches.doc = "number of creeping patches"
    stations = altar.properties.int(default=1)
    stations.doc = "number of surface observation locations"

    # dense output of slip rate
    n_eval = altar.properties.int(default=1)
    n_eval.doc = "number of t_eval points"
    t_eval_file = altar.properties.path(default="t_eval.txt")
    t_eval_file.doc = "the input file for time points when displacements are evaluated"

    #
    plate_loading_velocity = altar.properties.float(default=1)
    plate_loading_velocity.doc = "Plate loading velocity, as back slip"

    stress_kernel_file = altar.properties.path(default="stresskernel.txt")
    stress_kernel_file.doc = "the filename for input stress kernel - patches x patches matrix"

    stressrate_ext_file = altar.properties.path(default="stressrate_ext.txt")
    stressrate_ext_file.doc = (r"stress rate d\tau/dt imposed by external (locked) "
                               "patches - vector (patches)")

    displacement_kernel_file = altar.properties.path(default="displacementkernel.txt")
    displacement_kernel_file.doc = ("the filename for input displacement kernel G, arranged "
                                    "in (patches, stations)")

    # events - coseismic time and changes
    n_coseismic = altar.properties.int(default=2)
    n_coseismic.doc = ("number of the event/earthquake time within a cycle, "
                       "including start/end time")
    t_coseismic_file = altar.properties.path(default="t_coseismic.txt")
    t_coseismic_file.doc = "the input file for coseismic event time points"
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

    spin_up_max_cycles = altar.properties.int(default=5)
    spin_up_max_cycles.doc = "max number of cycles to stop spin up"

    # public data, use gPrefix to indicate cuda matrix/vector
    # keep track of (slip, velocity) at t_eval
    gT_eval = None
    gY_eval = None
    # coseismic events
    gTCoseismic = None
    gCoseismic = None
    # stress
    gStressKernel = None
    gStressRateExt = None

    gGF = None  # displacement kernel
    gSpinUpData = None
    gDataObsBatched = None  # data observations duplicated in #samples
    gDprediction = None
    # interface to C++ model object
    cmodel = None

    ode_solver = None  # to be implemented as a component in the future

    # protocol obligations
    @altar.export
    def initialize(self, application):
        """
        Initialize the state of the model given a {problem} specification
        """
        # chain up
        super().initialize(application=application)

        # load files to gpu
        self.loadInputFiles()

        # prepare the (slip, velocity) at t_eval, in order to be accessed from python
        self.gY_eval = altar.cuda.matrix(shape=(self.samples, self.n_eval*self.patches*2))

        # prepare the predicted data matrix
        self.gDprediction = altar.cuda.matrix(shape=(self.samples, self.observations),
                                              dtype=self.precision)

        # create the c model and pass parameters
        print("the model is run in precison", self.precision)
        if self.precision == "float32":  # single precision
            self.cmodel = libcudaseas.linearviscous.model_float()
        else:  # double precision
            self.cmodel = libcudaseas.linearviscous.model_double()

        # pass parameters and data to cmodel
        self.cmodel.initialize(
            self.samples, self.patches, self.stations,
            self.plate_loading_velocity,
            self.gStressKernel.data,
            self.gStressRateExt.data,
            self.gGF.data,
            self.nCoseismic, self.gTCoseismic, self.gCoseismic.data,
            self.t_eval_points, self.gT_eval.data, self.gY_eval.data,
            self.ode_solver_tolerance_absolute,
            self.ode_solver_tolerance_relative,
            self.self.spin_up_max_cycles
        )

        # set the initial state for spin up
        self.cmodel.set_spinup_data(
            self.gSpinUpData.data
        )
        # all done
        return self

    def loadInputFiles(self):
        """
        Load All Input files
        """
        # grab the report channels
        info = self.info
        error = self.error

        # grab the sizes
        patches = self.patches
        # stations = self.stations
        times = self.n_eval
        # max_samples = self.samples

        info.log(f'loading files from {self.case}...')
        # load the coseismic change to (slip, stress)
        self.gTCoseismic = self.loadFileToGPU(self.t_coseismic_file)
        self.nCoseismic = self.gTCoseismic.size

        self.gCoseismic = self.loadFileToGPU(self.coseismic_file)
        coseismic_shape = self.gCoseismic.shape
        if coseismic_shape != patches*2:
            error.log(f'Coseismic data shape {coseismic_shape} does not match 2*{patches}')

        # load the stress kernel (patches, patches)
        self.gStressKernel = self.loadFileToGPU(self.stress_kernel_file)
        sk_shape = self.gStressKernel.shape
        if sk_shape != (patches, patches):
            error.log(f'Stress kernel shape {sk_shape} does not match (patches, patches)')

        # load the stress rate imposed by external patches
        stressRateExt = self.loadFile(self.stressrate_ext_file)
        # multiply by the plate loading velocity
        stressRateExt *= -self.plate_loading_velocity
        # copy to gpu
        self.gStressRateExt = altar.cuda.vector(source=stressRateExt, dtype=self.precision)

        # init or load spin up data for (slips, velocities)
        if self.use_spin_up_data:
            self.gSpinUpData = self.loadFileToGPU(self.spin_up_data_file)
            spinup_data_shape = self.gSpinUpData.shape
            if spinup_data_shape != 2*patches:
                error.log(f'The spin up data shape {spinup_data_shape} does not match 2*{patches}')
        else:  # set as zeros
            self.gSpinUpData = altar.cuda.vector(shape=2*patches, dtype=self.precision).zero()

            # load the displacement kernel
        GF = self.loadFile(self.displacement_kernel_file)
        if GF.shape != (self.patches, self.stations):
            error.log(f'Displacement kernel shape {self.gTCoseismic.shape} does not '
                      f'match ({self.patches}, {self.stations}')

        # cd has been already merged to observed data through dataobs initialization
        # get a reference for the observed data (samples,
        self.gDataObsBatched = self.dataobs.gdataObsBatch
        # we now merge cd to GF
        cd = self.dataobs.gcd_inv.copy_to_host(type='numpy')
        bGF = self.mergeCdToGF(cd, GF)

        # copy it to gpu
        self.gGF = altar.cuda.matrix(source=bGF, dtype=self.precision)

        # load the t_eval points
        self.gT_eval = self.loadFileToGPU(self.t_eval_file)
        t_eval_shape = self.gT_eval.shape
        if t_eval_shape != times:
            error.log(f'the number of time points {t_eval_shape} does not match {times}')

        # debug
        # self.gSpinUpData.print()
        # self.gCoseismic.print()
        # self.gT_eval.print()
        # self.gStressKernel.print()
        # self.gGF.print()

        # all done
        return

    def mergeCdToGF(self, cd, GF):
        """
        Merge Data Covariance(cd) to GF (displacement kernel)
        @note that
        :param: cd - Data Covariance inverse in Cholesky decomposed form (obs, obs),
                     obs = times x stations
        :param: GF - Original displacement kernel (patches, stations)
        :return: bGF - merged GF (times, patchesxstations), the latter is flattened
        """

        # grab sizes
        patches = self.patches
        stations = self.stations
        times = self.t_eval_points

        # construct return
        bGF = numpy.zeros(shape=(times, patches, stations), dtype=self.precision)
        # iterate over different time points
        for time in range(times):
            # get a submatrix from cd for this time point
            # consider data at different time points are uncorrelated
            cd_t = cd[time*stations:(time+1)*stations, time*stations:(time+1)*stations]
            # multiply it by the original GF
            bGF[time, :, :] = numpy.matmul(GF, cd_t)
        # flatten the second dimension
        bGF = bGF.reshape(times, patches*stations)
        # all done
        return bGF

    def forwardModelBatched(self, theta, prediction, batch):
        """
        Linear Viscous forward model in batch
        :param theta: matrix (samples, parameters), sampling parameters
        :param prediction: matrix (samples, observations), the predicted data or residual
                           between predicted and observed data
        :param batch: integer, the number of samples to be computed batch<=samples
        :return: prediction as predicted data
        """

        parameters = theta.shape[1]
        self.cmodel.forward_model(theta.data, prediction.data, parameters, batch)

        # all done
        return prediction

    def cuEvalLikelihood(self, theta, likelihood, batch):
        """
        Compute the likelihood from my forward problem
        :param: theta - sampled parameters, matrix of (samples, parameters)
        :param: likelihood - computed likelihood, vector of (samples)
        :param: batch - number of samples to be computed
        """

        # get the data storage for data prediction or residual
        residuals = self.gDprediction

        # call forward model to calculate the data prediction or its difference between dataobs
        self.forwardModelBatched(theta=theta, prediction=residuals, batch=batch)

        # subtract from data observation
        residuals -= self.gDataObsBatched

        # call data method to calculate the l2 norm
        self.dataobs.cuEvalLikelihood(prediction=residuals, likelihood=likelihood,
                                      residual=True, batch=batch)
        # return the likelihood
        return likelihood

    @altar.export
    def forwardProblem(self, application, theta=None):
        """
        Perform the forward modeling with given {theta}
        """

        # all done
        return

    # private data
    # inputs

# end of file
