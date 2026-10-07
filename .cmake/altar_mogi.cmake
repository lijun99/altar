# -*- cmake -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# build the mogi package
function(altar_mogi_buildPackage)
  # install the sources straight from the source directory
  install(
    DIRECTORY mogi
    DESTINATION ${ALTAR_DEST_PACKAGES}/altar/models
    FILES_MATCHING PATTERN *.py
    )
  # build the package meta-data
  configure_file(
    mogi/meta.py.in mogi/meta.py
    @ONLY
    )
  # install the generated package meta-data file
  install(
    DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/mogi
    DESTINATION ${ALTAR_DEST_PACKAGES}/altar/models
    FILES_MATCHING PATTERN *.py
    )
  # all done
endfunction(altar_mogi_buildPackage)


# build the mogi extension module, with the forward model compiled in
function(altar_mogi_buildModule)
  # mogi
  Python_add_library(mogimodule MODULE)
  # adjust the name to match what python expects
  set_target_properties(
    mogimodule PROPERTIES
    LIBRARY_OUTPUT_NAME mogi
    SUFFIX ${PYTHON3_SUFFIX}
    )
  # set the include directories
  target_include_directories(
    mogimodule PRIVATE
    ${CMAKE_INSTALL_PREFIX}/include
    ${GSL_INCLUDE_DIRS}
    ${PYRE_INCLUDE_DIRS}
    )
  # set the libraries to link against
  target_link_libraries(mogimodule PRIVATE ${GSL_LIBRARIES} pybind11::module)
  # add the sources
  target_sources(mogimodule PRIVATE
    lib/libmogi/mogi.cc
    ext/mogi/mogi.cc
    )

  # install the mogi extension
  install(
    TARGETS mogimodule
    LIBRARY
    DESTINATION ${CMAKE_INSTALL_PREFIX}/packages/altar/models/mogi/ext
    )
endfunction(altar_mogi_buildModule)


# the scripts
function(altar_mogi_buildDriver)
  # install the scripts
  install(
    PROGRAMS bin/mogi
    DESTINATION bin
    )
  # all done
endfunction(altar_mogi_buildDriver)

# end of file
