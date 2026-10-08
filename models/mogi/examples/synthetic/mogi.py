#!/usr/bin/env python3
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
import numpy
from math import pi as π
# the framework
import altar
# my model
import altar.models.mogi


# app
class Mogi(altar.application, family="altar.applications.mogi"):
    """
    A generator of synthetic data for Mogi sources: writes the observation geometry to
    {geometry.csv}, the LOS displacements to {data.txt}, and their covariance to {cd.txt}
    """

    # user configurable state
    x = altar.properties.float(default=0)
    x.doc = "the x coördinate of the Mogi source"

    y = altar.properties.float(default=0)
    y.doc = "the y coördinate of the Mogi source"

    d = altar.properties.float(default=3000)
    d.doc = "the depth of the Mogi source"

    dV = altar.properties.float(default=1e7)
    dV.doc = "the volume change of the Mogi source, in m^3"

    nu = altar.properties.float(default=.25)
    nu.doc = "the Poisson ratio"

    offsets = altar.properties.list(schema=altar.properties.float(), default=[0, 0])
    offsets.doc = "the offsets of the eastern (oid 0) and western (oid 1) datasets"

    sigma = altar.properties.float(default=0.005)
    sigma.doc = "the standard deviation of the data noise, in m"

    noise = altar.properties.bool(default=False)
    noise.doc = "whether to add gaussian noise with {sigma} to the displacements"


    # protocol obligation
    @altar.export
    def main(self, *args, **kwds):
        """
        The main entry point
        """
        # the stations: a 1km grid
        stations = [(x*1000., y*1000.) for x in range(-5, 6) for y in range(-5, 6)]
        observations = len(stations)
        # observe all displacements from the same angle for now
        theta = π/4 # the incidence angle
        phi = π     # the azimuth, counterclockwise from east
        los = numpy.tile(
            [numpy.sin(theta) * numpy.cos(phi), numpy.sin(theta) * numpy.sin(phi), numpy.cos(theta)],
            (observations, 1))

        # compute the displacements
        source = altar.models.mogi.source(x=self.x, y=self.y, d=self.d, dV=self.dV, nu=self.nu)
        u = numpy.array(source.displacements(locations=stations, los=los))

        # the geometry; the western stations come from a different dataset
        geometry = altar.models.mogi.data(name="geometry")
        oid = numpy.zeros(observations, dtype=int)
        for idx, (x, y) in enumerate(stations):
            oid[idx] = 1 if x < 0 else 0
            record = geometry.pyre_new()
            record.oid = int(oid[idx])
            record.x = x
            record.y = y
            record.theta = theta
            record.phi = phi
        geometry.write(uri="geometry.csv")

        # the data, shifted by the dataset offsets
        u -= numpy.asarray(self.offsets)[oid]
        if self.noise:
            u += numpy.random.default_rng().normal(scale=self.sigma, size=observations)
        numpy.savetxt("data.txt", u)
        numpy.savetxt("cd.txt", self.sigma**2 * numpy.eye(observations))
        # all done
        return 0


# bootstrap
if __name__ == "__main__":
    # instantiate
    app = Mogi(name="mogi")
    # invoke
    status = app.run()
    # share
    raise SystemExit(status)


# end of file
