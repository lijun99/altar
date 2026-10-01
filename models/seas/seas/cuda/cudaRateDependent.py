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
from altar.models.seas.ext import cudaseas as libcudaseas
import numpy


# define model that only does the forward problem
class cudaRDModel():
    """
    Container class for the CUDA model implementation.
    """

    def __init__(self, precision, samples, patches, stations, plate_loading_velocity,
                 gStressKernel, gStressRateExt, gGF, nCoseismic, gTCoseismic, gCoseismic,
                 t_eval_points, gT_eval, gY_eval, ode_solver_tolerance_absolute,
                 ode_solver_tolerance_relative, spin_up_max_cycles, gSpinUpData):
        """
        Initialize a CUDA rate-dependent model object.
        """

        # create the c model and pass parameters
        print("the model is run in precison", precision)
        if precision == "float32":  # single precision
            self.cmodel = libcudaseas.ratedependent.model_float()
        else:  # double precision
            self.cmodel = libcudaseas.ratedependent.model_double()

        # pass parameters and data to cmodel
        self.cmodel.initialize(
            samples, patches, stations, plate_loading_velocity, gStressKernel,
            gStressRateExt, gGF, nCoseismic, gTCoseismic, gCoseismic,
            t_eval_points, gT_eval, gY_eval, ode_solver_tolerance_absolute,
            ode_solver_tolerance_relative, spin_up_max_cycles)

        # set the initial state for spin up
        self.cmodel.set_spinup_data(gSpinUpData)


# now define actual AlTar simulation that does forward and backward, using cudaRDModel
class cudaRateDependent(cudaBayesian, family="altar.models.seas.cuda.ratedependent"):
    """
    Creep model with rate-dependent rheology
    """

    # configurable traits

    # data observations
    # flattened 1d vector with time points x stations
    dataobs = altar.cuda.data.data()
    """ the observed data """
    dataobs.default = altar.cuda.data.datal2()
    dataobs.doc = dataobs.__doc__

    # model parameters
    # system information, each system consists of no. of patches and
    # 4 units (2 x slip, 2 x velocity) per patch
    patches = altar.properties.int(default=1)
    """number of creeping patches """
    patches.doc = patches.__doc__

    stations = altar.properties.int(default=1)
    """ number of surface observation locations """
    stations.doc = stations.__doc__

    # dense output of slip rate
    n_eval = altar.properties.int(default=1)
    """ number of t_eval points """
    n_eval.doc = n_eval.__doc__

    t_eval_file = altar.properties.path(default="t_eval.txt")
    """ the input file for time points when displacements are evaluated """
    t_eval_file.doc = t_eval_file.__doc__

    # fault information
    plate_loading_velocity = altar.properties.float(default=1)
    """ Plate loading velocity, as back slip """
    plate_loading_velocity.doc = plate_loading_velocity.__doc__

    # kernels
    stress_kernel_file = altar.properties.path(default="stresskernel.txt")
    """ the filename for input stress kernel - patches x patches matrix """
    stress_kernel_file.doc = stress_kernel_file.__doc__

    stressrate_ext_file = altar.properties.path(default="stressrate_ext.txt")
    r""" stress rate d\tau/dt imposed by external (locked) patches - vector (patches)"""
    stressrate_ext_file.doc = stressrate_ext_file.__doc__

    displacement_kernel_file = altar.properties.path(default="displacementkernel.txt")
    """ the filename for input displacement kernel G, arranged in (patches, stations) """
    displacement_kernel_file.doc = displacement_kernel_file.__doc__

    # events - coseismic time and changes
    n_coseismic = altar.properties.int(default=2)
    """ number of the event/earthquake time within a cycle, including start/end time """
    n_coseismic.doc = n_coseismic.__doc__

    t_coseismic_file = altar.properties.path(default="t_coseismic.txt")
    """ the input file for coseismic event time points """
    t_coseismic_file.doc = t_coseismic_file.__doc__

    coseismic_file = altar.properties.path(default="coseismic.txt")
    """ the input file for coseismic (slip, stress) changes, matrix with
    (events, 2*patches) elements """
    coseismic_file.doc = coseismic_file.__doc__

    use_spin_up_data = altar.properties.bool(default=False)
    """ whether to use a pre-computed data for (slip, velocity) at the end of an cycle """
    use_spin_up_data.doc = use_spin_up_data.__doc__

    spin_up_data_file = altar.properties.path(default="spin_up_data.txt")
    """ the input file for spin-up data of (slip, velocity) - 2*patches """
    spin_up_data_file.doc = spin_up_data_file.__doc__

    # solver settings
    ode_solver_tolerance_relative = altar.properties.float(default=1e-4)
    """ max relative error for ode solver """
    ode_solver_tolerance_relative.doc = ode_solver_tolerance_relative.__doc__

    ode_solver_tolerance_absolute = altar.properties.float(default=1e-3)
    """ max absolute error for ode solver """
    ode_solver_tolerance_absolute.doc = ode_solver_tolerance_absolute.__doc__

    spin_up_max_cycles = altar.properties.int(default=5)
    """ max number of cycles to stop spin up """
    spin_up_max_cycles.doc = spin_up_max_cycles.__doc__

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

    # interface to cudaRDModel object
    model = None
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

        # initialize the model object
        self.cmodel = cudaRDModel(
            precision=self.precision,
            samples=self.samples,
            patches=self.patches,
            stations=self.stations,
            plate_loading_velocity=self.plate_loading_velocity,
            gStressKernel=self.gStressKernel.data,
            gStressRateExt=self.gStressRateExt.data,
            gGF=self.gGF.data,
            nCoseismic=self.nCoseismic,
            gTCoseismic=self.gTCoseismic,
            gCoseismic=self.gCoseismic.data,
            t_eval_points=self.t_eval_points,
            gT_eval=self.gT_eval.data,
            gY_eval=self.gY_eval.data,
            ode_solver_tolerance_absolute=self.ode_solver_tolerance_absolute,
            ode_solver_tolerance_relative=self.ode_solver_tolerance_relative,
            spin_up_max_cycles=self.spin_up_max_cycles,
            gSpinUpData=self.gSpinUpData.data
            ).cmodel

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
