# -*- Makefile -*-
#
# michael a.g. aïvázis
# parasim
# (c) 1998-2020 all rights reserved
#

# the framework
include altar.def ${if ${value cuda.dir}, cudaaltar.def}

# models
include emhp.def gaussian.def mogi.def cdm.def linear.def ${if ${value cuda.dir}, cudalinear.def seismic.def}

# end of file
