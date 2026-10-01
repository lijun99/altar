# -*- cmake -*-
#
# Lijun Zhu (ljzhu@gps.caltech.edu)
# (c) 2019-present all rights reserved

# build the seas package
function(altar_seas_buildPackage)
  # install the sources straight from the source directory
  install(
    DIRECTORY seas
    DESTINATION ${ALTAR_DEST_PACKAGES}/altar/models
    FILES_MATCHING PATTERN *.py
    PATTERN seas/cuda EXCLUDE
    )
  # build the package meta-data
  configure_file(
    seas/meta.py.in seas/meta.py
    @ONLY
    )
  # install the generated package meta-data file
  install(
    DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/seas
    DESTINATION ${ALTAR_DEST_PACKAGES}/altar/models
    FILES_MATCHING PATTERN *.py
    )
  # all done
endfunction(altar_seas_buildPackage)


# build the seas cuda package
function(altar_seas_cuda_buildPackage)
  # install the sources straight from the source directory
  install(
    DIRECTORY seas/cuda
    DESTINATION ${ALTAR_DEST_PACKAGES}/altar/models/seas
    FILES_MATCHING PATTERN *.py
    )
  # all done
endfunction(altar_seas_cuda_buildPackage)


# build the seas extension module
function(altar_seas_cuda_buildModule)
  # seas; pybind11, like {altar_cuda_buildModule}
  Python_add_library(cudaseasmodule MODULE WITH_SOABI)
  # adjust the name to match what python expects
  set_target_properties(
    cudaseasmodule PROPERTIES
    LIBRARY_OUTPUT_NAME cudaseas
    )
  # set the include directories
  target_include_directories(
    cudaseasmodule PRIVATE
    ${CMAKE_INSTALL_PREFIX}/include
    ${PYRE_INCLUDE_DIRS}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
  # set the linker
  set_target_properties(cudaseasmodule PROPERTIES LINKER_LANGUAGE CUDA)
  # set  the link directories
  target_link_directories(
    cudaseasmodule PRIVATE
    ${CMAKE_INSTALL_PREFIX}/lib
    )
  # set the libraries to link against
  target_link_libraries(
    cudaseasmodule PRIVATE
    pybind11::module cudart cublas ${PYRE_LIBRARIES}
    )
  # add the sources
  target_sources(cudaseasmodule PRIVATE
    ext/cudaseas/cudaseas.cc
    ext/cudaseas/linearviscous/LinearViscous.cu
    ext/cudaseas/ratedependent/RateDependent.cu
    ext/cudaseas/tractiondependent/TractionDependent.cu
    ext/cudaseas/pyLinearViscous.cu
    ext/cudaseas/pyRateDependent.cu
    ext/cudaseas/pyTractionDependent.cu
    )

  # install the seas extension
  install(
    TARGETS cudaseasmodule
    LIBRARY
    DESTINATION ${CMAKE_INSTALL_PREFIX}/packages/altar/models/seas/ext
    )
endfunction(altar_seas_cuda_buildModule)


# the scripts
function(altar_seas_cuda_buildDriver)
  # install the scripts
  install(
    PROGRAMS bin/seas
    DESTINATION bin
    )
  # all done
endfunction(altar_seas_cuda_buildDriver)

# end of file
