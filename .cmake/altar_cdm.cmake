# -*- cmake -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# build the cdm package
function(altar_cdm_buildPackage)
  # install the sources straight from the source directory
  install(
    DIRECTORY cdm
    DESTINATION ${ALTAR_DEST_PACKAGES}/altar/models
    FILES_MATCHING PATTERN *.py
    )
  # build the package meta-data
  configure_file(
    cdm/meta.py.in cdm/meta.py
    @ONLY
    )
  # install the generated package meta-data file
  install(
    DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/cdm
    DESTINATION ${ALTAR_DEST_PACKAGES}/altar/models
    FILES_MATCHING PATTERN *.py
    )
  # all done
endfunction(altar_cdm_buildPackage)


# build the cdm extension module, with the forward model compiled in
function(altar_cdm_buildModule)
  # cdm
  Python_add_library(cdmmodule MODULE)
  # adjust the name to match what python expects
  set_target_properties(
    cdmmodule PROPERTIES
    LIBRARY_OUTPUT_NAME cdm
    SUFFIX ${PYTHON3_SUFFIX}
    )
  # set the include directories
  target_include_directories(
    cdmmodule PRIVATE
    ${CMAKE_INSTALL_PREFIX}/include
    ${GSL_INCLUDE_DIRS}
    ${PYRE_INCLUDE_DIRS}
    )
  # set the libraries to link against
  target_link_libraries(cdmmodule PRIVATE ${GSL_LIBRARIES} pybind11::module)
  # add the sources
  target_sources(cdmmodule PRIVATE
    lib/libcdm/cdm.cc
    ext/cdm/cdm.cc
    ext/cdm/bindings.cc
    )

  # install the cdm extension
  install(
    TARGETS cdmmodule
    LIBRARY
    DESTINATION ${CMAKE_INSTALL_PREFIX}/packages/altar/models/cdm/ext
    )
endfunction(altar_cdm_buildModule)


# the scripts
function(altar_cdm_buildDriver)
  # install the scripts
  install(
    PROGRAMS bin/altar-cdm
    DESTINATION bin
    )
  # all done
endfunction(altar_cdm_buildDriver)

# end of file
