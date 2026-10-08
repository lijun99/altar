# -*- cmake -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# build the altar package
function(altar_buildPackage)
  # install the sources straight from the source directory
  install(
    DIRECTORY altar
    DESTINATION ${ALTAR_DEST_PACKAGES}
    FILES_MATCHING PATTERN *.py
    )
  # build the package meta-data
  configure_file(
    altar/meta.py.in altar/meta.py
    @ONLY
    )
  # install the generated package meta-data file
  install(
    DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/altar
    DESTINATION ${ALTAR_DEST_PACKAGES}
    FILES_MATCHING PATTERN *.py
    )
  # all done
endfunction(altar_buildPackage)


# the scripts
function(altar_buildDriver)
  # install the scripts
  install(
    PROGRAMS bin/altar
    DESTINATION bin
    )
  # all done
endfunction(altar_buildDriver)

# end of file
