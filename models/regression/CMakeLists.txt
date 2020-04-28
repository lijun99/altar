# -*- cmake -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# (c) 2003-2020 all rights reserved

# get support
include(altar_regression)

# build the package
altar_regression_buildPackage()
# copy the driver
altar_regression_buildDriver()

# end of file
