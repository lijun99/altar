# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis (michael.aivazis@para-sim.com)
# grace bato           (mary.grace.p.bato@jpl.nasa.gov)
# eric m. gurrola      (eric.m.gurrola@jpl.nasa.gov)
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved


# externals
import numpy
# the package
import altar
# my base class, for its {psets}/{psets_list} and {dataobs} support
from altar.models.BayesianL2 import BayesianL2
# the layout of the observation geometry file
from .Data import Data as datasheet


# model declaration
class Reverso(BayesianL2, family="altar.models.reverso"):
    """
    An implementation of the Reverso 2-Magma Chamber Volcano Model, Reverso et al. [2014]

    The parameter sets, each with count 1, are {Qin}, the basal inflow rate, {H_s}, {a_s} and
    {H_d}, {a_d}, the depths and radii of the shallow and deep chambers, and {a_c}, the radius of
    the conduit between them; samples whose deep chamber isn't below the shallow one are
    rejected. The observed (east, north, up) displacements, three per observation, and their
    covariance are read by {dataobs}; {geometry} lists, in the same order, when and where, from
    the chambers, they were observed.
    """

    # user configurable state
    # the workspace name
    case = altar.properties.path(default="synthetic")
    case.doc = "the directory with the input files"

    geometry = altar.properties.path(default="geometry.csv")
    geometry.doc = "the csv file with the oid, t, x, y of each observation"

    # the computational strategy to use
    mode = altar.properties.str(default="fast")
    mode.doc = "the cpu implementation strategy: native (python) or fast (c++)"
    mode.validators = altar.constraints.isMember("native", "fast")

    # the shapes of the chambers
    shallow = altar.properties.str(default="sill")
    shallow.doc = "the shape of the shallow chamber: sill or sphere"
    shallow.validators = altar.constraints.isMember("sill", "sphere")

    deep = altar.properties.str(default="sill")
    deep.doc = "the shape of the deep chamber: sill or sphere"
    deep.validators = altar.constraints.isMember("sill", "sphere")

    # material parameters
    v = altar.properties.float(default=0.25)
    v.doc = "Poisson's ratio"

    mu = altar.properties.float(default=2000.0)
    mu.doc = "viscosity [Pa-s]"

    drho = altar.properties.float(default=300.0)
    drho.doc = "density difference (ρ_r-ρ_m), [kg/m**3]"

    # physical parameters
    G = altar.properties.float(default=20.0E9)
    G.doc = "shear modulus, [Pa, kg-m/s**2]"

    g = altar.properties.float(default=9.81)
    g.doc = "gravitational acceleration [m/s**2]"

    return_residual = altar.properties.bool(default=False)
    return_residual.doc = "the forward model returns residual(True) or prediction(False)"


    # framework obligations
    @altar.export
    def initialize(self, application):
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


    def verify_theta(self, theta, mask, batch=None):
        """
        Reject the samples outside the support of my priors, or whose deep chamber isn't below
        the shallow one
        """
        super().verify_theta(theta=theta, mask=mask, batch=batch)
        batch = theta.shape[0] if batch is None else batch
        θ = self.restrict(theta=theta)
        self._impl.verify(theta=θ, mask=mask, batch=batch)
        return mask


    def forward_model_batched(self, theta, prediction, batch=None):
        """
        Fill {prediction}, shape (samples x observations), with the predicted (east, north, up)
        displacements of each sample in {theta}
        """
        batch = theta.shape[0] if batch is None else batch
        return self._impl.forward_model_batched(theta=theta, prediction=prediction, batch=batch)


    @altar.export
    def forward_problem(self, application, theta):
        """
        The predicted displacements for each row of {theta}; see {altar.models.Model}
        """
        from .libreverso import REVERSO
        θ = numpy.atleast_2d(numpy.asarray(theta, dtype=float))
        t, x, y = self.stations.T
        prediction = numpy.empty((θ.shape[0], self.observations))
        for sample, parameters in enumerate(θ):
            u = REVERSO(t, x, y, **self.source(parameters), **self.medium())
            prediction[sample] = numpy.column_stack(u).ravel()
        return {"data": prediction}


    # implementation details
    def load_geometry(self):
        """
        Read the observation geometry into a (stations x 3) array of times and locations
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
        stations = numpy.array([(record.t, record.x, record.y) for record in sheet], dtype=float)
        # three displacements per station
        if 3 * len(stations) != self.observations:
            channel = self.error
            channel.log(f"'{self.geometry}' has {len(stations)} stations, so "
                        f"dataobs.observations should be {3*len(stations)}, not {self.observations}")
            raise SystemExit(1)
        return stations


    def find_layout(self):
        """
        The columns of (Qin, H_s, H_d, a_s, a_d, a_c) in the sample vector
        """
        layout = []
        for name in ("Qin", "H_s", "H_d", "a_s", "a_d", "a_c"):
            if name not in self.psets_list or self.psets[name].count != 1:
                channel = self.error
                channel.log(f"the reverso model needs a parameter set '{name}' with count=1")
                raise SystemExit(1)
            layout.append(self.psets[name].offset)
        return layout


    def source(self, parameters):
        """
        The model parameters of a sample, by name
        """
        names = ("Qin", "H_s", "H_d", "a_s", "a_d", "a_c")
        return dict(zip(names, (parameters[column] for column in self.layout)))


    def medium(self):
        """
        The material properties and the chamber shapes, by name
        """
        return dict(G=self.G, v=self.v, mu=self.mu, drho=self.drho, g=self.g,
                    shallow_sill=self.shallow == "sill", deep_sill=self.deep == "sill")


    def _makeImpl(self):
        """
        Build my implementation: cuda for the cuda backend, python or c++ on the cpu
        """
        if altar.backends.active() == "cuda":
            from .CUDA import CUDA as strategy
        elif self.mode == "native":
            from .Native import Native as strategy
        else:
            from .Fast import Fast as strategy
        return strategy()


    # private data
    stations = None # the times and locations of the observations, (stations x 3)
    layout = None # the columns of the model parameters in the sample vector
    _impl = None # my implementation strategy, chosen once, in {initialize}


# end of file
