
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# the framework
import altar
# externals
import numpy
from scipy.integrate import solve_ivp as ode_solver

# externals
from strikeslip import *


class LinearViscousSynthetic(altar.application, family="altar.shells.creepsynethic"):
    """
    Create a synthetic model for slip rate inversion
    assume the depth as length unit, slip rate normalized to its max value
    """

    # user configuration state
    patches = altar.properties.int(default=10)
    patches.doc = "number of patches for fault"

    asperity_patches = altar.properties.int(default=2)
    asperity_patches.doc = "number of asperity patches"

    fault_depth = altar.properties.float(default=100)
    fault_depth.doc = "the depth of all patches in km"

    vpl = altar.properties.float(default=1)
    vpl.doc = "plate loading velocity, as scale for slip rate cm/year"

    alpha_1 = altar.properties.float(default=0.1)

    spinup_periods = altar.properties.int(default=100)
    spinup_periods.doc = "number of periods used for spin up"

    spinup_tolerance = altar.properties.float(default=1e-4)
    spinup_tolerance.doc = "the error tolerance to end spin-up"

    t = altar.properties.array(default=(0, 1))
    t.doc = "time range for each period"

    stations = altar.properties.int(default=10)
    stations.doc = "number of observation stations (random generated)"

    obs_min = altar.properties.float(default=20)
    obs_min.doc  = "the min distance of station locations (random generated)"

    obs_max = altar.properties.float(default=200)
    obs_max.doc = "the max distance of stations"

    mu = altar.properties.float(default=1)
    mu.doc = "modulous"

    t_eval_points = altar.properties.int(default=10)
    t_eval_points.doc = "number of eval time points"


    # local parameters
    stressK = None
    stress_coseismic = None

    coseismic = None
    slip_cycle = None

    t_eval = None
    y_out = None
    G = None # displacement kernel


    # protocol obligation
    @altar.export
    def main(self, *args, **kwds):
        """
        The main entry point
        """
        channel = self.info
        channel.log(f'Generating synthetic model with {self.patches} patches and {self.stations} observations ...')

        # get number of patches
        patches = self.patches
        asperity_patches = self.asperity_patches
        self.creep_patches = patches - asperity_patches
        depth = self.fault_depth

        # generate stress kernel
        self.stressK = GenerateStressKernel(z0=0, z1=-self.fault_depth, patches = self.patches, mu=self.mu)
        self.stressK_creep = self.stressK[asperity_patches:, asperity_patches:]
        numpy.savetxt("stresskernel.txt", self.stressK_creep)

        # generate coseismic change of slip and velocity
        self.coseismic = self.generateCosesmicSlip()
        numpy.savetxt("coseismic.txt", self.coseismic)
        # for this simulation only
        self.coseismic[self.creep_patches:]/=self.alpha_1

        # generate stations and displacment kernel
        self.G = self.generate_stations() # (patches, stations)
        channel.log(f'Displace Kernel G shape (patches, stations) {self.G.shape}')


        tstart, tend = self.t
        self.t_eval = numpy.arange(1, self.t_eval_points+1)*(tend-tstart)/self.t_eval_points + tstart
        numpy.savetxt("t_eval.txt", self.t_eval)

        # simulation
        #slip_velocity = self.onecycle()
        slip_velocity = self.spinup()

        # take slips and rearrange in shape (times, patches)
        slip_t = numpy.transpose(slip_velocity[:self.creep_patches, :])

        channel.log(f'slip as functions of time shape (times, patches) {slip_t.shape}')
        # obtain surface displacement
        observation = numpy.matmul(slip_t, self.G)
        channel.log(f'observation shape (times, stations) {observation.shape}')
        # reshape it to 1d vector, (timesxstations)
        observation=observation.flatten(order='C')
        numpy.savetxt("displacements.txt", observation)
        channel.log(f'Done!')

        return

    def generateCosesmicSlip(self):

        import math

        patches = self.patches
        asperity = self.asperity_patches
        vpl = self.vpl
        tstart, tend = self.t
        span = tend-tstart
        self.slip_cycle = vpl*span
        slip_max = self.slip_cycle

        #penetration depth
        penetration_depth = 2

        slip = numpy.ndarray(shape=patches)
        stress = numpy.ndarray(shape=patches)


        for i in range(patches):
            if i<asperity :
                slip[i] = slip_max
            else:
                slip[i] = slip_max*math.exp(-(i-asperity+1)/penetration_depth)

        stressK = self.stressK
        stress = numpy.dot(slip, stressK)

        return numpy.concatenate((slip[asperity:], stress[asperity:]))

    def generate_stations(self):
        # generate random observation stations at (x, 0, 0)
        nstations = self.stations
        stations = numpy.zeros(shape=nstations)
        half = nstations//2
        stations[:half] = numpy.random.uniform(low=-self.obs_max, high=-self.obs_min, size=half)
        stations[half:] = numpy.random.uniform(low=self.obs_min, high=self.obs_max, size=nstations-half)
        # sort stations according to its x value
        stations = numpy.sort(stations)
        numpy.savetxt("stations.txt", stations)

        # generate the displacement kernel (Green's functions) shape=obs, patches
        GF = GenerateDisplacementKernel(z0=0, z1=-self.fault_depth, patches=self.patches, stations=stations)

        GF = numpy.transpose(GF[:, self.asperity_patches:])
        numpy.savetxt("displacementkernel.txt", GF)
        return GF

    def dydt(self, t, y):
        """
        compute dy/dt = f(t, y)
        :param t:
        :param y: y=(slip, velocity)
        :return:
        """
        # Get size
        patches = y.size//2
        alpha_1 = self.alpha_1
        vpl = self.vpl

        # Construct f for return
        f = numpy.ndarray(shape=y.shape, dtype=y.dtype)

        slip = y[:patches]
        velocity = y[patches:]-vpl

        # ds/dt
        f[:patches] = velocity

        # dv/dt
        stress = numpy.dot(velocity, self.stressK_creep)
        f[patches:] = stress/alpha_1

        return f


    def ode_solver_iterate(self, tspan, s0):

        fun = self.dydt
        result = ode_solver(fun, tspan, s0, method='RK45',
                   t_eval=None, dense_output=False, events=None, vectorized=False)
        t = result.t
        s1 = result.y

        return t, s1[:,-1]

    def ode_solver_final(self, tspan, s0, t_eval):

        fun = self.dydt
        result = ode_solver(fun, tspan, s0, method='RK45',
                   t_eval=t_eval, dense_output=True, events=None, vectorized=False)
        t = result.t
        s1 = result.y

        return t, s1

    def onecycle(self):
        """
        One cycle, to obtain the (slip, velocity) at given time points
        """
        s0 = numpy.zeros(2*self.creep_patches)
        s0 += self.coseismic

        t_span = self.t
        t_eval = self.t_eval
        t_out, y_out = self.ode_solver_final(t_span, s0, t_eval)
        return y_out

    def spinup(self):
        max_periods = self.spinup_periods
        tolerance = self.spinup_tolerance

        tspan = self.t
        t_period = tspan[1] -tspan[0]
        creep_patches = self.creep_patches


        coseismic = self.coseismic

        start = numpy.zeros(creep_patches*2)
        end = numpy.zeros(creep_patches*2)

        for period in range(max_periods):
            s0 = start + coseismic
            t, end = self.ode_solver_iterate(tspan, s0)

            print('cycle', period)
            print('slip', end[:creep_patches])
            print('velocity', end[creep_patches:])

            # temporary solution to override slip convergence
            end[:creep_patches] = 0
            diff = end-start
            print('difference', diff)
            start =  end
            tspan = tuple(t + t_period for t in tspan)
        # final step
        s0 = start+coseismic
        t_span = self.t
        t_eval = self.t_eval
        t_out, y_out = self.ode_solver_final(t_span, s0, t_eval)
        return y_out


# bootstrap
if __name__ == "__main__":
    # instantiate
    app = LinearViscousSynthetic(name="synthetic")
    # invoke
    status = app.run()
    # share
    raise SystemExit(status)



# end of file



