#!/usr/bin/env python3
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
# the framework
import altar
# my model
import altar.models.reverso


# the app
class Reverso(altar.application, family="altar.applications.reverso"):
    """
    A generator of synthetic data for {reverso} sources: writes the observation times and
    locations to {geometry.csv}, the (east, north, up) displacements to {data.txt}, and their
    covariance to {cd.txt}
    """

    # user configurable state
    H_s = altar.properties.float(default=3.0e3)
    H_s.doc = "depth of the shallow reservoir"

    H_d = altar.properties.float(default=4.0e3)
    H_d.doc = "depth of the deep reservoir"

    a_s = altar.properties.float(default=2.0e3)
    a_s.doc = "radius of the shallow magma reservoir"

    a_d = altar.properties.float(default=2.2e3)
    a_d.doc = "radius of the deep magma reservoir"

    a_c = altar.properties.float(default=1.5)
    a_c.doc = "radius of the hydraulic pipe connecting two magma reservoirs"

    Qin = altar.properties.float(default=0.6)
    Qin.doc = "basal magma inflow rate"

    G = altar.properties.float(default=20.0E9)
    G.doc = "shear modulus, [Pa, kg-m/s**2]"

    v = altar.properties.float(default=0.25)
    v.doc = "Poisson's ratio"

    mu = altar.properties.float(default=2000.0)
    mu.doc = "viscosity [Pa-s]"

    drho = altar.properties.float(default=300.0)
    drho.doc = "density difference (ρ_r-ρ_m), [kg/m**3]"

    g = altar.properties.float(default=9.81)
    g.doc = "gravitational acceleration [m/s**2]"


    # protocol obligations
    @altar.export
    def main(self, *args, **kwds):
        """
        The main entry point
        """
        # the observations: stations east of the chambers, from a microsecond to a year
        year = altar.units.time.year.value
        stations = numpy.array([(10**exponent * year, r, 0)
                                for exponent in range(-6, 1) for r in range(1000, 6000, 1000)],
                               dtype=float)
        t, x, y = stations.T

        geometry = altar.models.reverso.data(name="geometry")
        for ts, xs, ys in stations:
            record = geometry.pyre_new()
            record.oid = 0
            record.t = ts
            record.x = xs
            record.y = ys
        geometry.write(uri="geometry.csv")

        # the displacements, (east, north, up) for each observation
        u = numpy.column_stack(altar.models.reverso.source(
            t, x, y, Qin=self.Qin, H_s=self.H_s, H_d=self.H_d, a_s=self.a_s, a_d=self.a_d,
            a_c=self.a_c, G=self.G, v=self.v, mu=self.mu, drho=self.drho, g=self.g)).ravel()
        numpy.savetxt("data.txt", u)
        # 5% of the displacements, but no less than a centimeter
        numpy.savetxt("cd.txt", numpy.diag(numpy.maximum(0.05*numpy.abs(u), .01)**2))
        # all done
        return 0


# bootstrap
if __name__ == "__main__":
    # instantiate
    app = Reverso(name="reverso")
    # invoke
    status = app.run()
    # share
    raise SystemExit(status)


# end of file
