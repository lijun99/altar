# -*- cmake -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# (c) 2003-2022 all rights reserved

# get support
include(altar_seas)

# build the package
altar_seas_buildPackage()
# copy the driver
altar_seas_buildDriver()

# if CUDA is supported
if(WITH_CUDA)
  altar_seas_cuda_buildPackage()
  altar_seas_cuda_buildModule()
  altar_seas_cuda_buildDriver()
endif()
# end of file
