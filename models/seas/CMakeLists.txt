# -*- cmake -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# (c) 2003-2022 all rights reserved

# get support
include(altar_seas)

# all the seas models run on the gpu
if(WITH_CUDA)
  altar_seas_buildPackage()
  altar_seas_cuda_buildPackage()
  altar_seas_cuda_buildModule()
  altar_seas_cuda_buildDriver()
endif()

# end of file
