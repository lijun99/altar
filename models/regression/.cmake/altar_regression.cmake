# -*- cmake -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# (c) 2003-2020 all rights reserved

# build the regression package
function(altar_regression_buildPackage)
  # install the sources straight from the source directory
  install(
    DIRECTORY regression
    DESTINATION ${ALTAR_DEST_PACKAGES}/altar/models
    FILES_MATCHING PATTERN *.py
    )
  # build the package meta-data
  configure_file(
    regression/meta.py.in regression/meta.py
    @ONLY
    )
  # install the generated package meta-data file
  install(
    DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/regression
    DESTINATION ${ALTAR_DEST_PACKAGES}/altar/models
    FILES_MATCHING PATTERN *.py
    )
  # all done
endfunction(altar_regression_buildPackage)


# the scripts
function(altar_regression_buildDriver)
  # install the scripts
  install(
    PROGRAMS bin/regression
    DESTINATION bin
    )
  # all done
endfunction(altar_regression_buildDriver)

# end of file
