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
  # stage the mogi headers; the cuda library shares the point source formula
  altar_stageHeaders(lib/libmogi altar/models/mogi)
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


# build the mogi cuda library
function(altar_mogi_cuda_buildLibrary)
  # the libcudamogi target
  add_library(libcudamogi SHARED)
  # adjust the name
  set_target_properties(
    libcudamogi PROPERTIES
    LIBRARY_OUTPUT_NAME cudamogi
    )
  # set the include directories
  target_include_directories(
    libcudamogi PRIVATE
    ${CMAKE_INSTALL_PREFIX}/include
    ${GSL_INCLUDE_DIRS}
    ${PYRE_INCLUDE_DIRS}
    )
  # kernels index pyre grids directly; see {altar_cuda_buildLibrary}
  target_compile_definitions(libcudamogi PRIVATE WITH_CUDA)
  target_compile_options(libcudamogi PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  # add the sources
  target_sources(
    libcudamogi PRIVATE
    lib/libcudamogi/cudaMogi.cu
    )

  # stage the mogi cuda headers
  altar_stageHeaders(lib/libcudamogi altar/models/mogi/cuda)

  # install the library
  install(
    TARGETS libcudamogi
    LIBRARY DESTINATION lib
    )
  # all done
endfunction(altar_mogi_cuda_buildLibrary)


# build the mogi cuda extension module
function(altar_mogi_cuda_buildModule)
  # pybind11, like {altar_cuda_buildModule}
  Python_add_library(cudamogimodule MODULE WITH_SOABI)
  # adjust the name to match what python expects
  set_target_properties(
    cudamogimodule PROPERTIES
    LIBRARY_OUTPUT_NAME cudamogi
    LINKER_LANGUAGE CUDA
    )
  # set the include directories; altar/ext/cuda provides the shared binding helpers
  target_include_directories(
    cudamogimodule PRIVATE
    ${CMAKE_INSTALL_PREFIX}/include
    ${GSL_INCLUDE_DIRS}
    ${PYRE_INCLUDE_DIRS}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${CMAKE_CUDA_COMPILER_TOOLKIT_ROOT}/include/cccl
    ${CMAKE_SOURCE_DIR}/altar/ext/cuda
    )
  # set  the link directories
  target_link_directories(
    cudamogimodule PRIVATE
    ${CMAKE_INSTALL_PREFIX}/lib
    )
  # set the libraries to link against
  target_link_libraries(
    cudamogimodule PRIVATE
    libcudamogi pybind11::module cudart ${PYRE_LIBRARIES}
    )
  # add the sources
  target_sources(cudamogimodule PRIVATE
    ext/cudamogi/cudamogi.cc
    )

  # install the mogi cuda extension
  install(
    TARGETS cudamogimodule
    LIBRARY
    DESTINATION ${CMAKE_INSTALL_PREFIX}/packages/altar/models/mogi/ext
    )
endfunction(altar_mogi_cuda_buildModule)


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
