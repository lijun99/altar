# -*- cmake -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# (c) 2003-present all rights reserved

# get support
include(altar_gnss)

# build the package
altar_gnss_buildPackage()
# copy the driver
altar_gnss_buildDriver()

# end of file
