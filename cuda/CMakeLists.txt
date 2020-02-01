# -*- cmake -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# (c) 2003-2020 all rights reserved

# get support
include(altar_cudaframework)

# build the package
altar_cuda_buildPackage()
# build the library
altar_cuda_buildLibrary()
# build the extension
altar_cuda_buildModule()
# copy the driver
altar_cuda_buildDriver()

# end of file
