
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# the framework
import altar
# externals
import numpy

# externals
from strikeslip import *


class CreepSynthetic(altar.application, family="altar.shells.creepsynethic"):
    """
    Create a synthetic model for slip rate inversion
    assume the depth as length unit, slip rate normalized to its max value
    """

    # user configuration state
    patches = altar.properties.int(default=100)
    patches.doc = "number of patches for fault"

    observations = altar.properties.int(default=100)
    observations.doc = "number of observation stations (random generated)"

    vpl = altar.properties.float(default=1.0)
    vpl.doc = "plate loading velocity, as scale for slip rate"

    spinup_periods = altar.properties.int(default=20)

    t = altar.properties.array(default=(0, 1))


    # local parameters


    # protocol obligation
    @altar.export
    def main(self, *args, **kwds):
        """
        The main entry point
        """

        channel = self.info
        channel.log(f'Generating synthetic model with {self.patches} patches and {self.observations} observations ...')

        # get number of patches
        patches = self.patches

        # set the fault depth as unit
        depth = 1

        # generate stress kernel
        K = GenerateStressKernel(z0=0, z1=-depth, patches = patches, mu=1)
        numpy.savetxt("stresskernel.txt", K)

        # generate random observation stations at (x, 0, 0)
        observations = self.observations
        stations = numpy.zeros(shape=observations)
        half = observations//2
        stations[0:half] = numpy.random.uniform(low=-self.obs_max, high=-self.obs_min, size=half)
        stations[half:] = numpy.random.uniform(low=self.obs_min, high=self.obs_max, size=observations-half)
        # sort stations according to its x value
        stations = numpy.sort(stations)
        numpy.savetxt("stations.txt", stations)

        # generate the displacement kernel (Green's functions) shape=obs, patches
        G = GenerateDisplacementKernel(z0=0, z1=-depth, patches=patches, stations=stations)
        numpy.savetxt("displacementkernel.txt", G)

        # generate a synthetic distribution of sliprate
        v = self.GenerateSliprate(patches=patches, scale=self.vpl)
        numpy.savetxt("sliprate0.txt", v.reshape(patches,1))


        # set the output
        t_eval = numpy.arange(0, 1.05,0.05)


        channel.log(f'Done!')

        return


    def dydt(self, t, y, stress_kernel, alpha1):
        """
        compute dy/dt = f(t, y)
        :param t:
        :param y:
        :return:
        """
        # Get size
        size = y.size
        patches = size/2

        # Interpret y values
        slip = y[:patches]
        velocity = y[patches:]

        # Construct f values
        f = numpy.ndarray(shape=y.shape, dtype=y.dtype)
        dsdt = f[:patches]
        dvdt = f[patches:]

        # assign values
        dsdt[:] = velocity[:]
        #










    def ode_solver(self):


    def spinup(self):




# bootstrap
if __name__ == "__main__":
    # instantiate
    app = CreepSynthetic(name="synthetic.py")
    # invoke
    status = app.run()
    # share
    raise SystemExit(status)



# end of file



