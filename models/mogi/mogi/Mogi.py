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
from importlib.util import find_spec
import numpy
# the package
import altar
# my base class, for its {psets}/{psets_list} and {dataobs} support
from altar.models.BayesianL2 import BayesianL2
# the layout of the observation geometry file
from .Data import Data as datasheet


# declaration
class Mogi(BayesianL2, family="altar.models.mogi"):
    """
    An implementation of Mogi[1958]

    The surface displacement calculation for a pressure point source in an elastic half space.

    The parameter sets are {location} (x, y), {depth}, and {source}, the volume change dV, or
    log10(dV) with {log10_dV}; an optional {offsets} holds one shift per dataset, subtracted
    from the predicted LOS displacements of its observations. The observed LOS displacements
    and their covariance are read by {dataobs}; {geometry} lists, in the same order, where and
    along which LOS they were observed.
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

    log10_dV = altar.properties.bool(default=False)
    log10_dV.doc = "whether the {source} parameter is log10(dV) rather than dV, in m^3"

    # operating strategies
    mode = altar.properties.str(default="fast")
    mode.doc = "the cpu implementation strategy: native (python) or fast (c++)"
    mode.validators = altar.constraints.isMember("native", "fast")

    return_residual = altar.properties.bool(default=False)
    return_residual.doc = "the forward model returns residual(True) or prediction(False)"


    # protocol obligations
    @altar.export
    def initialize(self, application):
        """
        Initialize the state of the model given an {application} context
        """
        # chain up; mounts my input dataspace, loads the observations and lays out my psets
        super().initialize(application=application)
        # load the observation geometry and find my parameters in the sample vector
        self.stations = self.load_geometry()
        self.layout()
        # pick my implementation strategy
        self._impl = self._makeImpl()
        self._impl.initialize(model=self)
        # all done
        return self


    def forward_model_batched(self, theta, prediction, batch=None):
        """
        Fill {prediction}, shape (samples x observations), with the predicted LOS displacements
        of each sample in {theta}
        """
        batch = theta.shape[0] if batch is None else batch
        return self._impl.forward_model_batched(theta=theta, prediction=prediction, batch=batch)


    @altar.export
    def forward_problem(self, application, theta):
        """
        The predicted LOS displacements for each row of {theta}; see {altar.models.Model}
        """
        θ = numpy.atleast_2d(numpy.asarray(theta, dtype=float))
        stations = self.stations
        dV = 10**θ[:, self.sIdx] if self.log10_dV else θ[:, self.sIdx]
        # (samples, observations) offsets from the source
        x = stations[None, :, 0] - θ[:, self.xIdx, None]
        y = stations[None, :, 1] - θ[:, self.yIdx, None]
        d = θ[:, self.dIdx, None]
        R2 = x*x + y*y + d*d
        C = (1 - self.nu) * dV[:, None] / (numpy.pi * R2 * numpy.sqrt(R2))
        u = C * (x*stations[:, 2] + y*stations[:, 3] + d*stations[:, 4])
        # less the dataset offsets
        shifted = stations[:, 5] >= 0
        u[:, shifted] -= θ[:, stations[shifted, 5].astype(int)]
        return {"data": u}


    # implementation details
    def load_geometry(self):
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
        # filled in by {layout}, once the offsets are known
        stations[:, 5] = -1
        return stations


    def layout(self):
        """
        Record where my parameters live in the sample vector
        """
        psets = self.psets
        names = self.psets_list
        # check the required parameter sets
        for name, count in (("location", 2), ("depth", 1), ("source", 1)):
            if name not in names or psets[name].count != count:
                channel = self.error
                channel.log(f"the mogi model needs a parameter set '{name}' with count={count}")
                raise SystemExit(1)
        self.xIdx = psets["location"].offset
        self.yIdx = self.xIdx + 1
        self.dIdx = psets["depth"].offset
        self.sIdx = psets["source"].offset
        # the dataset offsets are optional
        if "offsets" in names:
            offsets = psets["offsets"]
            if self.oid.min() < 0 or self.oid.max() >= offsets.count:
                channel = self.error
                channel.log(f"oids in '{self.geometry}' must lie in [0, {offsets.count}), "
                            f"the count of the 'offsets' parameter set")
                raise SystemExit(1)
            self.stations[:, 5] = offsets.offset + self.oid
        # all done
        return


    def _makeImpl(self):
        """
        Build my implementation: cuda for the cuda backend, python or c++ on the cpu
        """
        if altar.backends.active() == "cuda":
            # the gpu kernels are cuTile's, an optional package
            if find_spec("cuda.tile") is None:
                self.error.log("the mogi gpu kernels need cuTile, pip install 'cuda-tile[tileiras]'; "
                               "or run on the cpu, with job.gpus = 0")
                raise SystemExit(1)
            from .CUDA import CUDA as strategy
        elif self.mode == "native":
            from .Native import Native as strategy
        else:
            from .Fast import Fast as strategy
        return strategy()


    # private data
    stations = None # the observation geometry, (observations x 6)
    oid = None # the dataset of each observation
    _impl = None # my implementation strategy, chosen once, in {initialize}


# end of file
