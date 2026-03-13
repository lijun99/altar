# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-2021 parasim inc
# (c) 2010-2021 california institute of technology
# all rights reserved
#
# Author(s): Lijun Zhu

# the package
import altar
import numpy
import numbers
from altar.models.BayesianL2 import BayesianL2


# declaration
class Static(BayesianL2, family="altar.models.seismic.static"):
    """
    Static inversion with CPU backend (d = G theta)
    Modeled as N patches with dip and slip displacements
    """

    # the number of patches
    patches = altar.properties.int(default=None)
    patches.doc = "the number of patches in the model"

    # the file based inputs
    green = altar.properties.path(default="static.gf.h5")
    green.doc = "the name of the file with the Green functions"

    # cpu forward-model hint for config parity with cuda
    use_tensor_core_gemm = altar.properties.bool(default=False)
    use_tensor_core_gemm.doc = "whether to use tensor core gemm for forward modeling"

    # options for performing forward model only
    forwardonly = altar.properties.bool(default=False)
    forwardonly.doc = "whether to run the simulation or the forward problem only"

    # input theta (one sample)
    theta_input = altar.properties.path(default="theta.h5")
    theta_input.doc = "the theta input file with a vector of parameters"

    theta_dataset = altar.properties.str(default=None)
    theta_dataset.doc = "the name/path of the theta dataset in h5 file"

    forward_output = altar.properties.path(default="forward_prediction.h5")
    forward_output.doc = "the name/path of the file to save forward problem results"

    # legacy output location for optional step dumps
    output_path = altar.properties.path(default="results")
    output_path.doc = "the output directory for saved steps"


    # protocol obligations
    @altar.export
    def initialize(self, application):
        """
        Initialize the state of the model given a {problem} specification
        """
        # chain up
        super().initialize(application=application)

        # load Green's function
        self.GF = self.load_file(
            filename=self.green,
            shape=(self.observations, self.parameters),
        )
        # keep a pristine copy for covariance updates
        self.GF0 = self.GF.clone()

        # compatibility aliases for Cp variants
        self.G = self.GF
        self.d = self.dataobs.dataobs
        self.Cd = self.dataobs.cd
        self.Cd_inv = getattr(self.dataobs, "cd_inv", None)
        self.normalization = getattr(self.dataobs, "normalization", None)
        # additional compatibility snapshots
        self.G0 = self.GF0
        if hasattr(self.d, "clone"):
            self.d0 = self.d.clone()
        if hasattr(self.Cd, "clone"):
            self.Cd0 = self.Cd.clone()

        # merge covariance to Green's function when the data is already scaled
        if not self.forwardonly and getattr(self.dataobs, "merge_cd_with_data", False):
            self.merge_covariance_to_gf()

        # all done
        return self


    def forward_model_batched(self, theta, prediction, green=None, batch=None, observation=None):
        """
        Linear forward model prediction = G * theta for a batch of samples.
        prediction has shape (samples, observations).
        """
        # resolve the inputs
        green = green or self.GF
        batch = batch or theta.rows

        # carve out working views if needed
        if batch == theta.rows:
            theta_view = theta
            pred_view = prediction
        else:
            theta_view = theta.view(start=(0, 0), shape=(batch, self.parameters))
            pred_view = prediction.view(start=(0, 0), shape=(batch, self.observations))

        # prediction = theta * green^T
        altar.blas.dgemm(
            theta_view.opNoTrans,
            green.opTrans,
            1.0,
            theta_view,
            green,
            0.0,
            pred_view,
        )

        # optionally convert prediction to residuals
        if observation is None and self.return_residual:
            observation = self.dataobs.dataobs

        if observation is not None:
            if hasattr(observation, "rows"):
                obs_view = observation
                if observation.rows != batch:
                    obs_view = observation.view(
                        start=(0, 0), shape=(batch, self.observations)
                    )
                pred_view -= obs_view
            else:
                for idx in range(batch):
                    row = pred_view.getRow(idx)
                    row -= observation

        # all done
        return self


    def forward_model(self, theta, green=None, prediction=None, observation=None):
        """
        Linear forward model prediction = G * theta for a single sample.
        """
        # resolve inputs
        green = green or self.GF
        if prediction is None:
            prediction = altar.vector(shape=self.observations)

        # prediction = G * theta, optionally subtract observation
        if observation is None:
            beta = 0.0
        else:
            prediction.copy(observation)
            beta = -1.0

        altar.blas.dgemv(green.opNoTrans, 1.0, green, theta, beta, prediction)

        # all done
        return prediction


    @altar.export
    def forward_problem(self, application, theta=None):
        """
        Perform the forward modeling with given {theta}
        """
        import h5py

        # load theta if not provided
        if theta is None:
            gtheta = self.load_file(
                filename=self.theta_input,
                shape=self.parameters,
                dataset=self.theta_dataset,
            )
        elif isinstance(theta, altar.vector):
            gtheta = theta
        else:
            gtheta = self._cpuToGsl(numpy.asarray(theta))

        # allocate predicted data
        data = altar.vector(shape=self.observations)
        # forward model (prediction only)
        self.forward_model(theta=gtheta, green=self.GF, prediction=data, observation=None)

        # save data prediction
        h5file = h5py.File(name=self.forward_output.path, mode='a')
        # if already exists, del the old dataset
        if 'static.Data' in h5file.keys():
            del h5file['static.Data']
        h5file.create_dataset(name='static.Data', data=numpy.asarray(data))
        h5file.close()

        # all done
        return


    def merge_covariance_to_gf(self):
        """
        Merge data covariance with Green's function when data is pre-scaled.
        """
        cd_inv = self.dataobs.cd_inv

        # reset to pristine Green's function before applying a new covariance
        if self.GF0 is not None:
            self.GF.copy(other=self.GF0)

        if isinstance(cd_inv, numbers.Number):
            self.GF *= cd_inv
        else:
            self.GF = altar.blas.dtrmm(
                cd_inv.sideLeft,
                cd_inv.upperTriangular,
                cd_inv.opNoTrans,
                cd_inv.nonUnitDiagonal,
                1.0,
                cd_inv,
                self.GF,
            )

        # keep compatibility aliases in sync
        self.G = self.GF

        # all done
        return self


    def load_file(self, filename, shape=None, dataset=None, dtype=None):
        """
        Load an input file to a gsl vector or matrix.
        Supported format:
        1. text file in '.txt' suffix, stored in prescribed shape
        2. binary file with '.bin' or '.dat' suffix
        3. hdf5 file in '.h5' suffix
        """
        dtype = dtype or numpy.float64

        ifs = self.ifs
        channel = self.error
        try:
            # get the path to the file
            file = ifs[filename]
        except ifs.NotFoundError:
            channel.log(f"missing input: no '{filename}' in '{ifs.path()}'")
            raise
        else:
            suffix = file.uri.suffix
            if suffix == '.txt':
                cpuData = numpy.loadtxt(file.uri.path, dtype=dtype)
            elif suffix in ('.bin', '.dat'):
                if shape is None:
                    raise channel.log(f"must specify shape for binary input '{filename}'")
                cpuData = numpy.fromfile(file.uri.path, dtype=dtype)
            elif suffix == '.h5':
                import h5py

                h5file = h5py.File(file.uri.path, 'r')
                if dataset is None:
                    dataset = list(h5file.keys())[0]
                cpuData = numpy.asarray(h5file.get(dataset), dtype=dtype)
                h5file.close()
            else:
                raise channel.log(f"unsupported input suffix '{suffix}' for '{filename}'")

        if shape is not None:
            cpuData = cpuData.reshape(shape)

        return self._cpuToGsl(cpuData)


    def _cpuToGsl(self, cpuData):
        """
        Convert a numpy array into a gsl vector or matrix.
        """
        if cpuData.ndim == 1:
            vec = altar.vector(shape=cpuData.shape[0])
            vec.ndarray()[:] = cpuData
            return vec
        if cpuData.ndim == 2:
            mat = altar.matrix(shape=cpuData.shape)
            mat.ndarray()[:] = cpuData
            return mat
        raise ValueError(f"unsupported data dimensions {cpuData.shape}")


    def compute_covariance_inverse(self, cd):
        """
        Compute the inverse of the data covariance matrix (compatibility helper).
        """
        # make a copy so we don't destroy the original
        cd = cd.clone()
        # perform the LU decomposition
        lu = altar.lapack.LU_decomposition(cd)
        # invert; this creates a new matrix
        inv = altar.lapack.LU_invert(*lu)
        # compute the Cholesky decomposition
        inv = altar.lapack.cholesky_decomposition(inv)

        # and return it
        return inv


    def compute_normalization(self, observations, cd):
        """
        Compute the normalization of the L2 norm (compatibility helper).
        """
        # support
        from math import log, pi
        # make a copy of cd
        cd = cd.clone()
        # compute its LU decomposition
        decomposition = altar.lapack.LU_decomposition(cd)
        # use it to compute the log of its determinant
        logdet = altar.lapack.LU_lndet(*decomposition)

        # all done
        return -(log(2 * pi) * observations + logdet) / 2


    def initialize_residuals(self, samples, data):
        """
        Initialize the residual matrix for compatibility with older workflows.
        """
        # allocate the residual matrix
        r = altar.matrix(shape=(data.shape, samples))
        # for each sample
        for sample in range(samples):
            # make the corresponding column a copy of the data vector
            r.setColumn(sample, data)
        # all done
        return r


    @altar.export
    def update(self, annealer):
        """
        Model updating at the bottom of each annealing step.
        """
        # get current worker
        worker = annealer.worker
        # check master
        if worker.rank == worker.manager:
            altar.utils.save_step(step=worker.step, path=self.output_path)
        # all done
        return self


    # private data
    GF = None
    GF0 = None


# end of file
