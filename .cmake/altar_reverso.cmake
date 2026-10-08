# -*- cmake -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# build the reverso package
function(altar_reverso_buildPackage)
  # install the sources straight from the source directory
  install(
    DIRECTORY reverso
    DESTINATION ${ALTAR_DEST_PACKAGES}/altar/models
    FILES_MATCHING PATTERN *.py
    )
  # build the package meta-data
  configure_file(
    reverso/meta.py.in reverso/meta.py
    @ONLY
    )
  # install the generated package meta-data file
  install(
    DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/reverso
    DESTINATION ${ALTAR_DEST_PACKAGES}/altar/models
    FILES_MATCHING PATTERN *.py
    )
  # all done
endfunction(altar_reverso_buildPackage)


# build the reverso extension module, with the forward model compiled in
function(altar_reverso_buildModule)
  # reverso
  Python_add_library(reversomodule MODULE)
  # adjust the name to match what python expects
  set_target_properties(
    reversomodule PROPERTIES
    LIBRARY_OUTPUT_NAME reverso
    SUFFIX ${PYTHON3_SUFFIX}
    )
  # set the include directories
  target_include_directories(
    reversomodule PRIVATE
    ${CMAKE_INSTALL_PREFIX}/include
    ${PYRE_INCLUDE_DIRS}
    )
  # set the libraries to link against
  target_link_libraries(reversomodule PRIVATE pybind11::module)
  # add the sources
  target_sources(reversomodule PRIVATE
    lib/libreverso/reverso.cc
    ext/reverso/reverso.cc
    ext/reverso/bindings.cc
    )

  # install the reverso extension
  install(
    TARGETS reversomodule
    LIBRARY
    DESTINATION ${CMAKE_INSTALL_PREFIX}/packages/altar/models/reverso/ext
    )
endfunction(altar_reverso_buildModule)


# the scripts
function(altar_reverso_buildDriver)
  # install the scripts
  install(
    PROGRAMS bin/altar-reverso
    DESTINATION bin
    )
  # all done
endfunction(altar_reverso_buildDriver)

# end of file
