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
# the framework
import altar
# my model
import altar.models.cdm


# app
class CDM(altar.application, family="altar.applications.cdm"):
    """
    A generator of synthetic data for CDM sources, seen by an ascending (oid 0) and a
    descending (oid 1) track: writes the observation geometry to {geometry.csv}, the LOS
    displacements to {data.txt}, and their covariance to {cd.txt}
    """

    # user configurable state
    x = altar.properties.float(default=0)
    x.doc = "the x coördinate of the CDM source"

    y = altar.properties.float(default=0)
    y.doc = "the y coördinate of the CDM source"

    d = altar.properties.float(default=3000)
    d.doc = "the depth of the CDM source"

    aX = altar.properties.float(default=1000)
    aX.doc = "the x semi-axis length"

    aY = altar.properties.float(default=800)
    aY.doc = "the y semi-axis length"

    aZ = altar.properties.float(default=600)
    aZ.doc = "the z semi-axis length"

    omegaX = altar.properties.float(default=10)
    omegaX.doc = "the CDM rotation about the x axis, in degrees"

    omegaY = altar.properties.float(default=-20)
    omegaY.doc = "the CDM rotation about the y axis, in degrees"

    omegaZ = altar.properties.float(default=30)
    omegaZ.doc = "the CDM rotation about the z axis, in degrees"

    opening = altar.properties.float(default=2)
    opening.doc = "the tensile component of the Burgers vector of the dislocation"

    nu = altar.properties.float(default=.25)
    nu.doc = "the Poisson ratio"

    offsets = altar.properties.list(schema=altar.properties.float(), default=[0, 0])
    offsets.doc = "the offsets of the ascending (oid 0) and descending (oid 1) datasets"

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
        x, y = numpy.meshgrid(numpy.arange(-6, 7)*1000., numpy.arange(-6, 7)*1000.)
        x = x.ravel()
        y = y.ravel()
        # the surface displacements
        ue, un, uv = altar.models.cdm.source(
            X=x, Y=y, X0=self.x, Y0=self.y, depth=self.d, opening=self.opening,
            ax=self.aX, ay=self.aY, az=self.aZ,
            omegaX=self.omegaX, omegaY=self.omegaY, omegaZ=self.omegaZ, nu=self.nu)

        # the two tracks: (incidence, azimuth counterclockwise from east) of the LOS vectors
        tracks = ((numpy.radians(35), numpy.radians(170)), (numpy.radians(40), numpy.radians(10)))
        geometry = altar.models.cdm.data(name="geometry")
        u = []
        for oid, (theta, phi) in enumerate(tracks):
            los = numpy.sin(theta)*numpy.cos(phi), numpy.sin(theta)*numpy.sin(phi), numpy.cos(theta)
            u.append(ue*los[0] + un*los[1] + uv*los[2] - self.offsets[oid])
            for xs, ys in zip(x, y):
                record = geometry.pyre_new()
                record.oid = oid
                record.x = xs
                record.y = ys
                record.theta = theta
                record.phi = phi
        geometry.write(uri="geometry.csv")

        # the data
        u = numpy.concatenate(u)
        if self.noise:
            u += numpy.random.default_rng().normal(scale=self.sigma, size=len(u))
        numpy.savetxt("data.txt", u)
        numpy.savetxt("cd.txt", self.sigma**2 * numpy.eye(len(u)))
        # all done
        return 0


# bootstrap
if __name__ == "__main__":
    # instantiate
    app = CDM(name="cdm")
    # invoke
    status = app.run()
    # share
    raise SystemExit(status)


# end of file
