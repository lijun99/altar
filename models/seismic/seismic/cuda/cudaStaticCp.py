# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-2019 parasim inc
# (c) 2010-2019 california institute of technology
# all rights reserved
#
# Author(s): Lijun Zhu


# the package
import altar
import altar.cuda
from altar.cuda import cublas
from altar.cuda import libcuda
from .cudaStatic import cudaStatic
import numpy

# declaration
class cudaStaticCp(cudaStatic, family="altar.models.seismic.cuda.staticcp"):
    """
    Static inversion with Cp (prediction error due to model parameter uncertainty)
    """

    # extra configurable traits for cp

    # mu is shear modulus; we use it for any generic model parameter
    nCmu = altar.properties.int(default=0)
    nCmu.doc = "the number of model parameters with uncertainties (or to be considered)"

    cmu_file = altar.properties.path(default="static.Cmu.th5")
    cmu_file.doc = "the covariance describing the uncertainty of model parameter, a nCmu x nCmu matrix"

    # kmu are a set sensitivity kernels (derivatives of Green's functions) with shape=(observations, parameters)
    kmu_file = altar.properties.path(default="static.kernel.h5")
    kmu_file.doc = "the sensitivity kernel of model parameters, a hdf5 file including nCmu kernel data sets"

    # initial model
    initial_model_file = altar.properties.path(default=None)
    initial_model_file.doc = "the initial mean model"

    beta_cp_start = altar.properties.float(default=0)
    beta_cp_start.doc = "for beta >= beta_cp_start, incorporate Cp into Cd"

    beta_use_initial_model = altar.properties.float(default=0)
    beta_use_initial_model.doc = "for beta <= beta_use_initial_model, use initial_model instead of mean model"

    dtype_cp = altar.properties.str(default='float32')
    dtype_cp.validators = altar.constraints.isMember('float32', 'float64')
    dtype_cp.doc = "single/double precision to compute Cp"

    # protocol obligations
    @altar.export
    def initialize(self, application):
        """
        Initialize the state of the model given a {problem} specification
        """
        # chain up
        # static model without Cp
        super().initialize(application=application)

        # initialize cp-specific parameters
        self.initializeCp()

        # all done
        return self

    def initializeCp(self):
        """
        Initialize Cp related
        :return:
        """
        # load Cmu
        self.gCmu = self.loadFileToGPU(filename=self.cmu_file, shape=(self.nCmu, self.nCmu), dtype=self.dtype_cp)
        # load initial model if provided
        if self.initial_model_file is not None:
            self.gInitModel = self.loadFileToGPU(filename=self.initial_model_file, shape=self.parameters, dtype=self.dtype_cp)
        # allocate a gpu vector to record mean model
        self.gMeanModel = altar.cuda.vector(shape=self.parameters, dtype=self.dtype_cp)
        # allocate a gpu matrix to record Cp
        self.Cp = altar.cuda.matrix(shape=(self.observations, self.observations), dtype=self.dtype_cp)
        return self


    def updateModel(self, annealer):
        """
        Model method called by Sampler before Metropolis sampling for each beta step starts,
        employed to compute Cp and merge Cp with data covariance
        :param annealer: the annealer for application
        :return: True or False if model parameters are updated or remain the same
        """
        # check beta and decide whether to incorporate cp
        step = annealer.worker.step
        beta = step.beta

        if beta < self.beta_cp_start:
            return False

        # get the threads information
        wid = annealer.worker.wid
        workers = annealer.worker.workers

        # calculate cp on master thread only
        # only master thread stores the mean_model for all samples (from previous beta step)
        if wid == 0:
            if beta <= self.beta_use_initial_model:
                # use initial(input) model
                mean_model = self.gInitModel
            else:
                # copy the current mean model from all samples
                self.gMeanModel.copy_from_host(source=step.mean)
                # use the mean model
                mean_model = self.gMeanModel

            # compute Cp with mean model
            self.computeCp(model=mean_model, cp=self.Cp)

        # if more than one workers, bcast Cp
        if workers > 1:
            self.Cp.bcast(communicator=annealer.worker.communicator, source=0)

        # recompute covariance = cp + cd,
        # and merge covariance with observed data
        self.dataobs.updateCovariance(cp=self.Cp)
        # merge covariance with green's function
        self.mergeCovarianceToGF()

        # all done
        return True

    def computeCp(self, model, cp=None):
        """
        Compute Cp with a mean model
        :param model:
        :return:
        """

        # force float64 computation
        import h5py
        import numpy

        # grab the samples  shape=(samples, parameters)
        parameters = self.parameters
        observations = self.observations
        nCmu = self.nCmu

        dtype_cp = 'float64'
        dtype_model = model.dtype

        # allocate Cp if not pre-allocated
        Cp = cp or altar.cuda.matrix(shape=(observations, observations), dtype=self.dtype_cp)

        # get cmu, shape=(nCmu, nCmu); kmu are loaded on the fly
        Cmu = self.gCmu

        # allocate work arrays
        kmu = altar.cuda.matrix(shape=self.gGF.shape, dtype=self.dtype_cp)
        Kp = altar.cuda.matrix(shape=(nCmu, observations), dtype = self.dtype_cp)
        kpv = altar.cuda.vector(shape=observations, dtype=Kp.dtype)

        # check the existence of kernel h5 file
        h5kernelfile = self.ifs[self.kmu_file]
        # open h5 file
        h5kernel = h5py.File(h5kernelfile.uri.path, 'r')
        # get the keys for datasets (kernels)
        h5keys =list(h5kernel.keys())

        for i in range(nCmu):
            # load kmu_np(cpu) from h5, shape=(observations, parameters)
            kmu_np = numpy.asarray(h5kernel.get(h5keys[i]), dtype=self.dtype_cp).reshape(kmu.shape)
            # copy it gpu
            kmu.copy_from_host(source=kmu_np)
            # call the forward model
            self.forwardModel(theta=model, green=kmu, prediction=kpv)
            # copy the vector result to matrix
            Kp.set_row(kpv, row=i)


        # KpC = Cm Kp (nCmu, obs) = (nCmu, nCmu)x(nCmu, obs)
        KpC = cublas.symm(A=Cmu, B=Kp, handle=self.cublas_handle)

        # Cp = Kp^T KpC (obs, obs) = (obs, nCmu) x (nCmu, obs)
        Cp = cublas.gemm(A=KpC, transa= cublas.OpTrans, B=Kp, transb = cublas.OpNoTrans,
                         out =Cp, handle = self.cublas_handle)

        #libcuda.cublas_gemm(self.cublas_handle,
        #                    0, 1, # transa, transb
        #                    Kp.shape[1], Cp.shape[1], Kp.shape[0], # m, n, k
        #                    1.0,   # alpha
        #                    Kp.data, Kp.shape[1], # A, lda
        #                    KpC.data, KpC.shape[1], # B, ldb
        #                    0.0,
        #                    Cp.data, Cp.shape[1])

        # close the kernel h5 file
        h5kernel.close()

        # all done
        return Cp


    # private data
    gCmu = None
    gInitModel = None
    gMeanModel = None





# end of file
