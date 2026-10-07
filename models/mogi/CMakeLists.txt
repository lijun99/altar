# -*- cmake -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# (c) 2003-present all rights reserved

# get support
include(altar_mogi)

# build the package
altar_mogi_buildPackage()
# build the extension
altar_mogi_buildModule()
# copy the driver
altar_mogi_buildDriver()

# if CUDA is supported
if(WITH_CUDA)
  altar_mogi_cuda_buildLibrary()
  altar_mogi_cuda_buildModule()
endif()

# end of file
