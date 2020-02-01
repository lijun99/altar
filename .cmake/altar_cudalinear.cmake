# -*- cmake -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# (c) 2003-2020 all rights reserved

# build the cudalinear package
function(altar_cudalinear_buildPackage)
  # install the sources straight from the source directory
  install(
    DIRECTORY cudalinear
    DESTINATION ${ALTAR_DEST_PACKAGES}/altar/models
    FILES_MATCHING PATTERN *.py
    )
  # build the package meta-data
  configure_file(
    cudalinear/meta.py.in cudalinear/meta.py
    @ONLY
    )
  # install the generated package meta-data file
  install(
    DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/cudalinear
    DESTINATION ${ALTAR_DEST_PACKAGES}/altar/models
    FILES_MATCHING PATTERN *.py
    )
  # all done
endfunction(altar_cudalinear_buildPackage)


# the scripts
function(altar_cudalinear_buildDriver)
  # install the scripts
  install(
    PROGRAMS bin/cudalinear
    DESTINATION bin
    )
  # all done
endfunction(altar_cudalinear_buildDriver)

# end of file
