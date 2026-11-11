# -*- cmake -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# build the gnss package
function(altar_gnss_buildPackage)
  # install the sources straight from the source directory
  install(
    DIRECTORY gnss
    DESTINATION ${ALTAR_DEST_PACKAGES}/altar/models
    FILES_MATCHING PATTERN *.py
    )
  # build the package meta-data
  configure_file(
    gnss/meta.py.in gnss/meta.py
    @ONLY
    )
  # install the generated package meta-data file
  install(
    DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/gnss
    DESTINATION ${ALTAR_DEST_PACKAGES}/altar/models
    FILES_MATCHING PATTERN *.py
    )
  # all done
endfunction(altar_gnss_buildPackage)


# the scripts
function(altar_gnss_buildDriver)
  # install the scripts
  install(
    PROGRAMS bin/altar-gnss
    DESTINATION ${ALTAR_DEST_BIN}
    )
  # all done
endfunction(altar_gnss_buildDriver)

# end of file
