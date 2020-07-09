# -*- cmake -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# (c) 2003-2020 all rights reserved

# build the sir package
function(altar_sir_buildPackage)
  # install the sources straight from the source directory
  install(
    DIRECTORY sir
    DESTINATION ${ALTAR_DEST_PACKAGES}/altar/models
    FILES_MATCHING PATTERN *.py
    )
  # build the package meta-data
  configure_file(
    sir/meta.py.in sir/meta.py
    @ONLY
    )
  # install the generated package meta-data file
  install(
    DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/sir
    DESTINATION ${ALTAR_DEST_PACKAGES}/altar/models
    FILES_MATCHING PATTERN *.py
    )
  # all done
endfunction(altar_sir_buildPackage)


# the scripts
function(altar_sir_buildDriver)
  # install the scripts
  install(
    PROGRAMS bin/SIR
    DESTINATION bin
    )
  # all done
endfunction(altar_sir_buildDriver)

# end of file
