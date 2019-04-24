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
    
    area_patches_file = altar.properties.path(default=None)
    area_patches_file.doc = "input file for area of each patch, in unit of km^2"
    
    area = altar.properties.float(default=1.0)
    area.doc = "total area in unit of km^2" 
    
    Mw_mean = altar.properties.float(default=1.0)
    Mw_mean.doc = " the mean moment magnitude scale"
    
    Mw_sigma = altar.properties.float(default=0.5)
    Mw_sigma.doc = " the variance of moment magnitude scale"
    
    Mu = altar.properties.float(default = 32)
    Mu.doc = "the shear modulus in unit of GPa"
    
    support = altar.properties.array(default=(0,1))
    support.doc = "the support interval of the prior distribution"
    
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

        # assign the rng
        self.rng = application.rng.rng
        
        # initialize the area for each patches
        self.patches = self.parameters  
        # by default, assign the constant patch_area to each patch 
        self.area_patches = altar.vector(shape=self.patches).fill(self.area)
        # if a file is provided, load it 
        if self.area_patches_file is not None: 
            self.area_patches.load(self.area_patches_file.uri)
        
    def cuInitSample(self, theta):
        """
        Fill my portion of {theta} with initial random values from my distribution.
        """
        # use cpu to generate a batch of samples
        samples = theta.shape[0]
        parameters = self.parameters
        θ = altar.matrix(shape=(samples, parameters))
    
        # grab the area of patches
        area_patches = self.area_patches
        
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
            # Pentiar = M0/Mu =  \sum (A_i D_i) 
            # 15 here is for GPa * Km^2, instead of Pa * m^2   
            Pentier = pow(10, 1.5*Mw + 9.1 - 15)/self.Mu 
            # generate a dirichlet sample \sum x_i = 1
            dirichlet_D.vector(vector=theta_sample)
            # D_i = P * x_i /A_i 
            for parameter in range (parameters):
                theta_sample[parameter]*=Pentier/area_patches[parameter]
            # set theta 
            θ.setRow(sample, theta_sample)

        #mean,sd=θ.mean_sd(axis=0)
        #mean.print()
        #sd.print()

        # make a copy to gpu
        gθ = altar.cuda.matrix(source=θ, dtype=self.precision)
        # insert into theta according to the idx_range
        theta.insert(src=gθ, start=(0,self.idx_range[0]))

        # and return
        return self

    # private member variables
    area_patches = None
    patches = None
    rng = None

# end of file
