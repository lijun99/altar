# -*- cmake -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# (c) 2003-2020 all rights reserved

# get support
include(altar_seismic)

# build the package
altar_seismic_buildPackage()
# build the library
# altar_seismic_buildLibrary()
# build the extension
# altar_seismic_buildModule()
# copy the driver
altar_seismic_buildDriver()

# if CUDA is supported
if(WITH_CUDA)
  altar_seismic_cuda_buildPackage()
  altar_seismic_cuda_buildLibrary()
  altar_seismic_cuda_buildModule()
  altar_seismic_cuda_buildDriver()
endif()
# end of file
