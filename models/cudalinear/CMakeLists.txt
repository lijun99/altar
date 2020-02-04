# -*- cmake -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# (c) 2003-2020 all rights reserved

# get support
include(altar_cudalinear)

# only cuda version is available
if(WITH_CUDA)
  # build the package
  altar_cudalinear_buildPackage()
  # copy the driver
  altar_cudalinear_buildDriver()
endif()
# end of file
