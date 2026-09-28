# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# the package
import altar
import altar.cuda

import numpy


# the declaration
class DataL2:
    """
    The cuda implementation of observed data with L2 norm

    My configuration (data_file, observations, cd_file, cd_std, merge_cd_with_data, norm) is
    copied down from the shim, once, before {initialize} runs. Unlike the cpu side, I always
    merge the data covariance into the observed data ({dataobs_batch}), regardless of
    {merge_cd_with_data} -- this mirrors the pre-port {cudaDataL2}, which had its own
    always-True `merge_cd_to_data` and never exposed the choice as a trait; unifying that with
    the cpu default (False) is a separate, deferred decision, not made here.
    """

    @altar.export
    def initialize(self, application):
        """
        Initialize data obs from model
        """
        self.precision = application.job.gpuprecision
        self.cd_dtype = self.cd_dtype or self.precision

        self.ifs = application.pfs["inputs"]
        self.error = application.error
        self.info = application.info
        self.samples = application.job.chains

        observations = self.observations

        self.dataobs = self.load_file(filename=self.data_file, shape=observations)

        if self.cd_file is not None:
            self.cd = self.load_file(
                filename=self.cd_file, shape=(observations, observations), dtype=self.cd_dtype)
        else:
            # a constant variance needs no dense matrix unless a {cp} is added to it later
            self.cd = None

        self.initialize_covariance()
        # all done
        return self


    def eval_likelihood(self, prediction, likelihood, residual=True, batch=None):
        """
        compute the datalikelihood for prediction

        {prediction}: (samples x observations) grid of predicted data
        {likelihood}: (samples,) grid, pre-allocated
        {residual}: whether {prediction} is already subtracted by the observed data
        {batch}: number of (first few) samples to be computed
        """
        batch = batch or prediction.shape[0]

        # depending on convenience, users may
        # either copy dataobs to their model and use the residual as input of prediction
        # or compute prediction from forward model and subtract the dataobs here
        if not residual:
            pred = numpy.asarray(prediction)
            pred[:batch, :] -= numpy.asarray(self.dataobs_batch)[:batch, :]

        # the data covariance is always pre-merged into {prediction} on the gpu path (see the
        # class docstring), so no {sigma_inv} is passed here
        self.norm.eval_likelihood(
            v=prediction, constant=self.normalization, batch=batch, out=likelihood)
        # all done
        return likelihood


    @property
    def dataobs_batch(self):
        """
        A batch of duplicated observations, one copy per sample
        """
        return self._dataobs_batch


    def release_cd(self):
        """
        Release {cd_inv} from gpu memory
        """
        self.cd_inv = None
        return


    def load_file(self, filename, shape, dataset=None, dtype=None):
        """
        Load an input file to a numpy array (for both float32/64 support)

        Supported formats: a '.txt' text file in the prescribed shape; a '.bin'/'.dat' binary
        file at the desired precision, reshaped per {shape}; or (preferred) a '.h5' file
        carrying its own shape/precision metadata
        """
        dtype = dtype or self.precision

        ifs = self.ifs
        channel = self.error
        try:
            file = ifs[filename]
        except not ifs.NotFoundError:
            channel.log(f"no file '{filename}' found in '{ifs.path()}'")
            raise
        else:
            suffix = file.uri.suffix
            if suffix == '.txt':
                cpuData = numpy.loadtxt(file.uri.path, dtype=dtype).reshape(shape)
            elif suffix == '.bin' or suffix == '.dat':
                cpuData = numpy.fromfile(file.uri.path, dtype=dtype).reshape(shape)
            elif suffix == '.h5':
                import h5py
                h5file = h5py.File(file.uri.path, 'r')
                if dataset is None:
                    dataset = list(h5file.keys())[0]
                cpuData = numpy.asarray(h5file.get(dataset), dtype=dtype).reshape(shape)
                h5file.close()
        # all done
        return cpuData


    def initialize_covariance(self):
        """
        Initialize gpu data and data covariance
        """
        observations = self.observations
        samples = self.samples

        self._dataobs_batch = pyre_grid_managed(shape=(samples, observations), cell=self.precision)

        self.update_covariance()
        # all done
        return self


    def update_covariance(self, cp=None):
        """
        Update the data covariance C_chi = Cd + Cp, then refresh everything derived from it:
        its inverse (in Cholesky-decomposed form), the l2 normalization, and the merged data

        {cp}: an (observations x observations) grid, the model-uncertainty contribution to
        add to Cd; {None} to just (re)use Cd alone
        """
        from math import log, pi as π

        # a constant variance, cd_std^2: no factorization needed
        if cp is None and self.cd is None:
            return self._constant_covariance()

        cusolver = altar.cuda.cusolver
        cublas = altar.cuda.cublas
        handle = altar.cuda.cusolver_handle()

        observations = self.observations
        double = self.cd_dtype == "float64"
        potrf = cusolver.dpotrf if double else cusolver.spotrf
        potrf_buffer_size = cusolver.dpotrf_buffer_size if double else cusolver.spotrf_buffer_size
        potri = cusolver.dpotri if double else cusolver.spotri
        potri_buffer_size = cusolver.dpotri_buffer_size if double else cusolver.spotri_buffer_size

        gCchi = pyre_grid_managed(shape=(observations, observations), cell=self.cd_dtype)
        if self.cd is None:
            numpy.asarray(gCchi)[:, :] = 0
            numpy.fill_diagonal(numpy.asarray(gCchi), self.cd_std ** 2)
        else:
            numpy.asarray(gCchi)[:, :] = self.cd
        self._chi_variance = None
        self._covariance = None
        if cp is not None:
            cp_arr = numpy.asarray(cp).astype(self.cd_dtype, copy=False)
            numpy.asarray(gCchi)[:, :] += cp_arr
            self._chi_variance = numpy.diag(numpy.asarray(gCchi)).astype(float)
            self._covariance = numpy.array(gCchi, dtype=float)

        devInfo = pyre_grid_managed(shape=(1,), cell="int32")

        # {potrf}/{potri} are column-major; passing {uplo=LOWER} throughout and reading the
        # buffer back row-major (numpy's own convention) gives, at each stage, the *upper*
        # triangle of the row-major-interpreted matrix -- a row-major buffer read as
        # column-major is its own transpose, so a col-major lower triangle is a row-major
        # upper triangle of the same data. This is what lets the same {uplo} flag be reused
        # unchanged across all three calls below, and lands the final factor in the row-major
        # upper triangle -- the convention {altar.norms.cuda.L2._apply_covariance} and the
        # cpu {DataL2.initialize_covariance} (`Cd_inv.upperTriangular`) both expect.
        def _factor(A):
            lwork = potrf_buffer_size(handle, cublas.FillMode.LOWER, observations, A, observations)
            workspace = pyre_grid_managed(shape=(max(lwork, 1),), cell=self.cd_dtype)
            potrf(handle, cublas.FillMode.LOWER, observations, A, observations, workspace, lwork, devInfo)

        # factor Cchi = U^T U (U in the row-major upper triangle); the factorization fails,
        # and says where, iff Cchi is not positive definite
        _factor(gCchi)
        if int(numpy.asarray(devInfo)[0]) != 0:
            self.error.log(
                f"the data covariance C_chi is not positive definite: its Cholesky "
                f"factorization failed at row {int(numpy.asarray(devInfo)[0])}")
            raise SystemExit(1)
        # invert it in place, from that factor: gCchi now holds Cd_inv's row-major upper
        # triangle (Cd_inv is symmetric, so only one triangle is meaningful)
        lwork = potri_buffer_size(handle, cublas.FillMode.LOWER, observations, gCchi, observations)
        workspace = pyre_grid_managed(shape=(max(lwork, 1),), cell=self.cd_dtype)
        potri(handle, cublas.FillMode.LOWER, observations, gCchi, observations, workspace, lwork, devInfo)
        # re-factor: gCchi now holds Cd_inv = U^T U, U (Cholesky factor of Cd_inv) in the
        # row-major upper triangle
        _factor(gCchi)

        # the log determinant of Cd_inv's Cholesky factor is half the log determinant of
        # Cd_inv itself; the same normalization the cpu path computes
        logdet = numpy.log(numpy.diag(numpy.asarray(gCchi))).sum()
        self.normalization = -0.5 * log(2 * π) * observations + logdet

        # keep a copy of the factor at the working precision (cd_dtype may differ, to improve
        # accuracy of the covariance computation itself)
        if self.cd_dtype == self.precision:
            self.cd_inv = gCchi
        else:
            self.cd_inv = pyre_grid_managed(shape=(observations, observations), cell=self.precision)
            numpy.asarray(self.cd_inv)[:, :] = numpy.asarray(gCchi)

        # load the observed data and merge the covariance into it
        gDataVec = pyre_grid_managed(shape=(observations,), cell=self.precision)
        numpy.asarray(gDataVec)[:] = self.dataobs
        gDataVec = self.merge_cdto_data(cd_inv=self.cd_inv, data=gDataVec)

        # duplicate it into a (samples x observations) batch
        numpy.asarray(self._dataobs_batch)[:, :] = numpy.asarray(gDataVec)[None, :]
        # all done
        return self


    def observed(self):
        """
        The raw observed data, as a numpy vector
        """
        return numpy.asarray(self.dataobs, dtype=float)


    def sigma(self):
        """
        The standard deviation of each observation, as a numpy vector
        """
        if self.cd is None:
            return numpy.full(self.observations, float(self.cd_std))
        return numpy.sqrt(numpy.diag(self.cd)).astype(float)


    def sigma_chi(self):
        """
        The standard deviation of each observation under C_chi, as a numpy vector
        """
        return self.sigma() if self._chi_variance is None else numpy.sqrt(self._chi_variance)


    def covariance(self):
        """
        The covariance in effect, C_d or C_chi = C_d + C_p: a numpy (observations x observations)
        array, or a float, the common variance, when it is a constant times the identity
        """
        if self._covariance is not None:
            return self._covariance
        if self.cd is None:
            return float(self.cd_std) ** 2
        return numpy.array(self.cd, dtype=float)


    def _constant_covariance(self):
        """
        Cd = cd_std^2 I: {cd_inv} is the scalar 1/cd_std, the factor of Cd_inv = cd_inv^2 I
        """
        from math import log, pi as π
        observations = self.observations
        self._chi_variance = None
        self._covariance = None
        self.cd_inv = 1.0 / self.cd_std
        self.normalization = -0.5 * log(2 * π) * observations - observations * log(self.cd_std)
        numpy.asarray(self._dataobs_batch)[:, :] = (numpy.asarray(self.dataobs) * self.cd_inv)[None, :]
        return self


    def check_positive_definiteness(self, matrix, name=None):
        """
        Check the positive definiteness of a gpu (observations x observations) grid

        Managed memory is host-visible directly, so a zero-copy {numpy.asarray} is all
        reading it back for the eigenvalue check takes -- no explicit host copy needed
        """
        name = name or 'Matrix'
        cm = numpy.asarray(matrix)
        evals = numpy.linalg.eigvalsh(cm)
        minval = evals.min()
        maxval = evals.max()

        if minval <= 0:
            self.error.log(
                f"{name} is not positive definite, with eigenvalues from {minval} to "
                f"{maxval}. Aborting ...")
            raise SystemExit(1)

        if minval / maxval < 1.e-6:
            self.info.log(
                f"Warning: for {name}, the ratio between min and max eigenvalues are too "
                f"small, which may cause convergence issues")
        # all done
        return self


    def merge_cdto_data(self, cd_inv, data):
        """
        Merge the data covariance matrix into the observed data

        {cd_inv}: the inverse of the covariance matrix, Cholesky-decomposed, with its factor
        U (Cd_inv = U^T U) in the row-major upper triangle
        {data}: the raw observed data, a (observations,) grid

        Returns {data <- U @ data}, a fresh grid
        """
        cublas = altar.cuda.cublas
        handle = altar.cuda.cublas_handle()
        double = self._cell(data) == "float64"
        trmv = cublas.dtrmv if double else cublas.strmv

        n = data.shape[0]
        gDataVec = pyre_grid_managed(shape=(n,), cell=self._cell(data))
        numpy.asarray(gDataVec)[:] = numpy.asarray(data)

        # {cd_inv}'s factor U lives in the row-major upper triangle; cublas is column-major,
        # so the same "opposite triangle" translation used to build {cd_inv} applies here
        # too: passing {uplo=LOWER, transa=T} reads that row-major-upper U and computes
        # x <- U @ x
        trmv(
            handle,
            cublas.FillMode.LOWER, cublas.Operation.T, cublas.DiagType.NON_UNIT,
            n, cd_inv, n, gDataVec, 1,
        )
        # all done
        return gDataVec


    @staticmethod
    def _cell(grid):
        """
        The cell type name (e.g. "float64") a pyre.grid.Grid was built with
        """
        format = memoryview(grid).format
        return "float64" if format == "d" else "float32"


    # configuration, copied down from the shim by {_makeImpl}
    data_file = None
    observations = None
    cd_file = None
    cd_std = None
    merge_cd_with_data = None
    # diag(C_chi), when a C_p is part of it
    _chi_variance = None
    _covariance = None # C_chi, once a C_p is added
    norm = None
    cd_dtype = None

    # local variables
    normalization = 0
    ifs = None
    error = None
    info = None
    samples = None
    precision = None
    dataobs = None
    cd = None
    cd_inv = None
    _dataobs_batch = None


def pyre_grid_managed(shape, cell):
    """
    Allocate a fresh grid of cuda managed memory; a thin indirection so this module doesn't
    need a hard import of {pyre.cuda} at module-load time before cuda is known to be active
    """
    import pyre.cuda
    return pyre.cuda.managed(shape=shape, cell=cell)


# end of file
