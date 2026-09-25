# -*- Makefile -*-
#
# michael a.g. aïvázis
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#
# parasim

# the framework
include altar.def ${if ${value cuda.dir}, cudaaltar.def}

# models
include emhp.def gaussian.def mogi.def cdm.def linear.def ${if ${value cuda.dir}, cudalinear.def seismic.def}

# end of file
