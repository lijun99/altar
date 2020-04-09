# -*- python -*-
# -*- coding: utf-8 -*-
#
# lijun zhu (ljzhu@gps.caltech.edu)
#
# (c) 2013-2018 parasim inc
# (c) 2010-2018 california institute of technology
# all rights reserved
#

# get the package
import altar
import altar.cuda

# get the protocol

# and my base class
from altar.cuda.distributions.cudaUniform import cudaUniform

# the declaration
class cudaMoment(cudaUniform, family="altar.cuda.distributions.moment"):
    """
    The probability distribution for displacements (D) conforming to a given Moment magnitude scale
    Mw = (log M0 - 9.1)/1.5 (Hiroo Kanamori)
    M0 = Mu A D
    It serves to initialize samples only, with combined gaussian and dirichlet distributions.
    It inherits uniform distribution for verification and density calculations.
    """

    # user configurable state
    # patches = altar.properties.int(default=1)
    # patches.doc = "number of patches"
    # The value of patches is provided by parameters

    area_patch_file = altar.properties.path(default=None)
    area_patch_file.doc = "input file for area of each patch, in unit of km^2"

    area = altar.properties.array(default=[1.0])
    area.doc = "area of each patch in unit of km^2, provide one value if the same for all patches"

    Mw_mean = altar.properties.float(default=1.0)
    Mw_mean.doc = " the mean moment magnitude scale"

    Mw_sigma = altar.properties.float(default=0.5)
    Mw_sigma.doc = " the variance of moment magnitude scale"

    Mu = altar.properties.array(default = [32])
    Mu.doc = "the shear modulus for each patch in GPa, provide one value if the same for all patches"

    Mu_patch_file = altar.properties.path(default=None)
    Mu_patch_file = "input file for the shear modulus of each patch, in Km^2"

    slip_sign = altar.properties.str(default='positive')
    slip_sign.validators = altar.constraints.isMember("positive", "negative")
    slip_sign.doc = "the sign of slips, all positive or all negative"

    # protocol obligations
    @altar.export
    def initialize(self, application):
        """
        Initialize with the given random number generator
        """
        # all done
        return self

    def cuInitialize(self, application):
        """
        cuda interface of initialization
        """
        # initialize the parent uniform distribution
        super().cuInitialize(application=application)

        # get the input path
        ifs = application.pfs["inputs"]

        # assign the rng
        self.rng = application.rng.rng

        # set the number of patches
        self.patches = self.parameters
        # initialize the area for each patch
        if len(self.area) == 1:
            # by default, assign the constant patch_area to each patch
            self.area_patches = altar.vector(shape=self.patches).fill(self.area[0])
        elif len(self.area) != self.patches:
            # if the size doesn't match
            channel = self.error
            raise channel.log("the size of area doesn't match the number of patches")
        else:
            #
            self.area_patches = self.area

        # if a file is provided, load it
        if self.area_patch_file is not None:
            try:
                # get the path to the file
                areafile = ifs[self.area_patch_file]
            # if the file doesn't exist
            except ifs.NotFoundError:
                # grab my error channel
                channel = self.error
                # complain
                channel.log(f"missing area_patch_file: no '{self.area_patch_file}' {ifs.path()}")
                # and raise the exception again
                raise
            # if all goes well
            else:
                # allocate the vector
                self.area_patches = altar.vector(shape=self.patches)
                # and load the file contents into memory
                self.area_patches.load(self.areafile.uri)

        # initialize the shear modulus for each patch
        if len(self.Mu) == 1:
            # by default, assign the constant to each patch
            self.mu_patches = altar.vector(shape=self.patches).fill(self.Mu[0])
        elif len(self.Mu) != self.patches:
            # if the size doesn't match
            channel = self.error
            raise channel.log("the size of Mu doesn't match the number of patches")
        else:
            #
            self.mu_patches = self.Mu



        # all done
        return self



    def cuInitSample(self, theta):
        """
        Fill my portion of {theta} with initial random values from my distribution.
        """
        # use cpu to generate a batch of samples
        samples = theta.shape[0]
        parameters = self.parameters
        θ = altar.matrix(shape=(samples, parameters))

        # grab the references for area/shear modulus
        area_patches = self.area_patches
        mu_patches = self.mu_patches

        # create a gaussian distribution to generate Mw for each sample
        gaussian_Mw = altar.pdf.gaussian(mean=self.Mw_mean, sigma=self.Mw_sigma, rng=self.rng)

        # create a dirichlet distribution to generate displacements
        alpha = altar.vector(shape=parameters).fill(1) # power 0, or (alpha_i = 1)
        dirichlet_D = altar.pdf.dirichlet(alpha=alpha, rng=self.rng)

        # create a tempory vector for theta of samples
        theta_sample = altar.vector(shape=parameters)
        # iterate through samples to initialize samples
        for sample in range(samples):
            # generate a Mw sample
            Mw = gaussian_Mw.sample()
            # Pentiar = M0 =  \sum (A_i D_i Mu_i)
            # 15 here is for GPa * Km^2, instead of Pa * m^2
            Pentier = pow(10, 1.5*Mw + 9.1 - 15)
            # if a negative sign is desired
            if self.slip_sign == 'negative':
                Pentier = - Pentier
            # generate a dirichlet sample \sum x_i = 1
            dirichlet_D.vector(vector=theta_sample)
            # D_i = P * x_i /A_i
            for patch in range(parameters):
                theta_sample[patch]*=Pentier/(area_patches[patch]*mu_patches[patch])
            # set theta
            θ.setRow(sample, theta_sample)

        # make a copy to gpu
        gθ = altar.cuda.matrix(source=θ, dtype=self.precision)
        # insert into theta according to the idx_range
        theta.insert(src=gθ, start=(0,self.idx_range[0]))

        # and return
        return self

    # private member variables
    area_patches = None
    mu_patches = None
    patches = None
    rng = None

# end of file
