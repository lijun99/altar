# -*- python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#


# externals
from __future__ import annotations
from importlib.util import find_spec
import typing
import numpy
# the package
import altar
# my base class, for its {psets}/{psets_list} and {dataobs} support
from altar.models.BayesianL2 import BayesianL2
# the layout of the observation geometry file
from .Data import Data as datasheet

if typing.TYPE_CHECKING:
    from altar.arrays import Array
    from altar.bayesian.states.BayesianState import BayesianState
    from altar.shells.Application import Application


# declaration
class CDM(BayesianL2, family="altar.models.cdm"):
    """
    The compound dislocation model of Nikkhoo et al. [2017]: three mutually orthogonal
    rectangular dislocations with a common opening, in an elastic half space

    The parameter sets are {location} (x, y), {depth}, {opening}, the semi-axes, as {a} with
    count 3 or {aX}, {aY}, {aZ}, and the rotation angles about the x, y and z axes in degrees,
    as {omega} with count 3 or {omegaX}, {omegaY}, {omegaZ}; an optional {offsets} holds one
    shift per dataset, subtracted from the predicted LOS displacements of its observations.
    Samples whose source reaches above the free surface are rejected. The observed LOS
    displacements and their covariance are read by {dataobs}; {geometry} lists, in the same
    order, where and along which LOS they were observed.
    """


    # user configurable state
    # the name of the test case
    case = altar.properties.path(default="synthetic")
    case.doc = "the directory with the input files"

    geometry = altar.properties.path(default="geometry.csv")
    geometry.doc = "the csv file with the oid, x, y, theta, phi of each observation"

    # the material properties
    nu = altar.properties.float(default=.25)
    nu.doc = "the Poisson ratio"

    # operating strategies
    mode = altar.properties.str(default="fast")
    mode.doc = "the cpu implementation strategy: native (python) or fast (c++)"
    mode.validators = altar.constraints.isMember("native", "fast")

    return_residual = altar.properties.bool(default=False)
    return_residual.doc = "the forward model returns residual(True) or prediction(False)"


    # protocol obligations
    @altar.export
    def initialize(self, application: Application) -> typing.Self:
        """
        Initialize the state of the model given an {application} context
        """
        # chain up; mounts my input dataspace, loads the observations and lays out my psets
        super().initialize(application=application)
        # load the observation geometry and find my parameters in the sample vector
        self.stations = self.load_geometry()
        self.layout = self.find_layout()
        # pick my implementation strategy
        self._impl = self._makeImpl()
        self._impl.initialize(model=self)
        # all done
        return self


    @altar.export
    def initialize_sample(self, step: BayesianState, batch: int | None = None) -> typing.Self:
        """
        Draw the initial sample from my prior, replacing the sources that reach above the free
        surface with copies of the ones that don't
        """
        super().initialize_sample(step=step, batch=batch)
        samples = step.theta.shape[0] if batch is None else batch
        # find the invalid ones
        if altar.backends.active() == "cuda":
            mask = altar.cuda.vector(shape=step.theta.shape[0], dtype="int32").zero()
        else:
            mask = numpy.zeros(step.theta.shape[0])
        self.verify_theta(theta=step.theta, mask=mask, batch=samples)
        if altar.backends.active() == "cuda":
            altar.cuda.synchronize()
            invalid = numpy.flatnonzero(numpy.asarray(mask.grid)[:samples])
        else:
            invalid = numpy.flatnonzero(mask[:samples])
        if len(invalid) == 0:
            return self
        valid = numpy.setdiff1d(numpy.arange(samples), invalid)
        if len(valid) == 0:
            channel = self.error
            channel.log("every source drawn from the prior reaches above the free surface")
            raise SystemExit(1)
        # replace them, in physical and, if separate, sampling space
        sources = self.rng.rng.choice(valid, size=len(invalid))
        buffers = [step.theta]
        if self.has_reparametrization:
            buffers.append(step.theta_sampling)
        for buffer in buffers:
            θ = numpy.asarray(buffer.grid) if altar.backends.active() == "cuda" else buffer
            θ[invalid] = θ[sources]
        # all done
        return self


    def verify_theta(self, theta: Array, mask: Array, batch: int | None = None) -> Array:
        """
        Reject the samples outside the support of my priors, or whose source reaches above the
        free surface
        """
        super().verify_theta(theta=theta, mask=mask, batch=batch)
        batch = theta.shape[0] if batch is None else batch
        θ = self.restrict(theta=theta)
        self._impl.verify(theta=θ, mask=mask, batch=batch)
        return mask


    def forward_model_batched(self, theta: Array, prediction: Array,
                              batch: int | None = None) -> typing.Self:
        """
        Fill {prediction}, shape (samples x observations), with the predicted LOS displacements
        of each sample in {theta}
        """
        batch = theta.shape[0] if batch is None else batch
        return self._impl.forward_model_batched(theta=theta, prediction=prediction, batch=batch)


    @altar.export
    def forward_problem(self, application: Application, theta: numpy.ndarray) -> dict:
        """
        The predicted LOS displacements for each row of {theta}; see {altar.models.Model}
        """
        from .ext import libcdm
        θ = numpy.atleast_2d(numpy.asarray(theta, dtype=float))
        samples = θ.shape[0]
        prediction = numpy.zeros((samples, self.observations))
        libcdm.displacements(θ, numpy.ascontiguousarray(self.stations, dtype=numpy.float64),
                             self.layout, self.nu, samples, prediction)
        return {"data": prediction}


    # implementation details
    def load_geometry(self) -> numpy.ndarray:
        """
        Read the observation geometry into a (observations x 6) array of the location, the LOS
        unit vector (east, north, up) and the column of the dataset offset (-1 for none)
        """
        # get the file
        try:
            node = self.ifs[self.geometry]
        except self.ifs.NotFoundError:
            channel = self.error
            channel.log(f"missing observation geometry: no '{self.geometry}' in '{self.case}'")
            raise
        # read it
        sheet = datasheet(name="geometry")
        sheet.read(uri=node.uri)
        records = list(sheet)
        # it must match the observations
        if len(records) != self.observations:
            channel = self.error
            channel.log(f"'{self.geometry}' has {len(records)} observations, "
                        f"but dataobs.observations is {self.observations}")
            raise SystemExit(1)

        stations = numpy.empty((len(records), 6))
        self.oid = numpy.array([record.oid for record in records], dtype=int)
        for obs, record in enumerate(records):
            # the LOS unit vector from the ground to the observing craft
            stations[obs, :5] = (record.x, record.y,
                                 numpy.sin(record.theta) * numpy.cos(record.phi),
                                 numpy.sin(record.theta) * numpy.sin(record.phi),
                                 numpy.cos(record.theta))
        # the dataset offsets are optional
        stations[:, 5] = -1
        if "offsets" in self.psets_list:
            offsets = self.psets["offsets"]
            if self.oid.min() < 0 or self.oid.max() >= offsets.count:
                channel = self.error
                channel.log(f"oids in '{self.geometry}' must lie in [0, {offsets.count}), "
                            f"the count of the 'offsets' parameter set")
                raise SystemExit(1)
            stations[:, 5] = offsets.offset + self.oid
        return stations


    def find_layout(self) -> list[int]:
        """
        The columns of (x0, y0, depth, opening, ax, ay, az, omegaX, omegaY, omegaZ) in the
        sample vector
        """
        psets = self.psets
        names = self.psets_list

        def column(name, count=1):
            if name not in names or psets[name].count != count:
                return None
            return psets[name].offset

        def triplet(name):
            # either one set of three, or three sets of one
            start = column(name, count=3)
            if start is not None:
                return [start, start + 1, start + 2]
            columns = [column(name + axis) for axis in "XYZ"]
            if None in columns:
                channel = self.error
                channel.log(f"the cdm model needs a parameter set '{name}' with count=3, "
                            f"or '{name}X', '{name}Y', '{name}Z' with count=1")
                raise SystemExit(1)
            return columns

        layout = []
        for name, count in (("location", 2), ("depth", 1), ("opening", 1)):
            start = column(name, count)
            if start is None:
                channel = self.error
                channel.log(f"the cdm model needs a parameter set '{name}' with count={count}")
                raise SystemExit(1)
            layout.extend(range(start, start + count))
        layout.extend(triplet("a"))
        layout.extend(triplet("omega"))
        return layout


    def _makeImpl(self) -> typing.Any:
        """
        Build my implementation: cuda for the cuda backend, python or c++ on the cpu
        """
        if altar.backends.active() == "cuda":
            # the gpu kernels are cuTile's, an optional package
            if find_spec("cuda.tile") is None:
                self.error.log("the cdm gpu kernels need cuTile, pip install 'cuda-tile[tileiras]'; "
                               "or run on the cpu, with job.gpus = 0")
                raise SystemExit(1)
            from .CUDA import CUDA as strategy
        elif self.mode == "native":
            from .Native import Native as strategy
        else:
            from .Fast import Fast as strategy
        return strategy()


    # private data
    stations: numpy.ndarray | None = None # the observation geometry, (observations x 6)
    oid: numpy.ndarray | None = None # the dataset of each observation
    layout: list[int] | None = None # the columns of the source parameters in the sample vector
    _impl: typing.Any = None # my implementation strategy, chosen once, in {initialize}


# end of file
