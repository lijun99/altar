# -*- cmake -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# build the seismic package
function(altar_seismic_buildPackage)
  # install the sources straight from the source directory
  install(
    DIRECTORY seismic
    DESTINATION ${ALTAR_DEST_PACKAGES}/altar/models
    FILES_MATCHING PATTERN *.py
    PATTERN seismic/cuda EXCLUDE
    )
  # build the package meta-data
  configure_file(
    seismic/meta.py.in seismic/meta.py
    @ONLY
    )
  # install the generated package meta-data file
  install(
    DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/seismic
    DESTINATION ${ALTAR_DEST_PACKAGES}/altar/models
    FILES_MATCHING PATTERN *.py
    )
  # all done
endfunction(altar_seismic_buildPackage)

# the scripts
function(altar_seismic_buildDriver)
  # install the scripts
  install(
    PROGRAMS bin/seismic
    DESTINATION bin
    )
  # all done
endfunction(altar_seismic_buildDriver)

# build the seismic cuda package
function(altar_seismic_cuda_buildPackage)
  # install the sources straight from the source directory
  install(
    DIRECTORY seismic/cuda
    DESTINATION ${ALTAR_DEST_PACKAGES}/altar/models/seismic
    FILES_MATCHING PATTERN *.py
    )
  # all done
endfunction(altar_seismic_cuda_buildPackage)


# buld the seismic cuda libraries
function(altar_seismic_cuda_buildLibrary)
  # the libcudaseismic target
  add_library(libcudaseismic SHARED)
  # adjust the name
  set_target_properties(
    libcudaseismic PROPERTIES
    LIBRARY_OUTPUT_NAME seismic
    )
  # set the include directories
  target_include_directories(
    libcudaseismic PRIVATE
    ${CMAKE_INSTALL_PREFIX}/include
    ${Python3_NumPy_INCLUDE_DIRS}
    ${PYRE_INCLUDE_DIRS}
    )
  # set the link directories
  target_link_directories(
    libcudaseismic PRIVATE
    ${CMAKE_INSTALL_PREFIX}/lib
    ${PYRE_PREFIX_PATH}/lib
    )
  # add the dependencies
  target_link_libraries(
    libcudaseismic PRIVATE
    ${PYRE_LIBRARIES} cublas
    )
  # kernels index pyre grids directly; see {altar_cuda_buildLibrary}
  target_compile_definitions(libcudaseismic PRIVATE WITH_CUDA)
  target_compile_options(libcudaseismic PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  # add the sources
  target_sources(
    libcudaseismic PRIVATE
    lib/libcudaseismic/cudaKinematic_kernels.cu
    lib/libcudaseismic/cudaKinematic.cu
    lib/libcudaseismic/cudaMoment.cu
    lib/libcudaseismic/version.cc
    )

  # stage the seismic headers
  altar_stageHeaders(lib/libcudaseismic altar/models/seismic/cuda)

  # install the library
  install(
    TARGETS libcudaseismic
    LIBRARY DESTINATION lib
    )

  # all done
endfunction(altar_seismic_cuda_buildLibrary)


# build the seismic extension module
function(altar_seismic_cuda_buildModule)
  # seismic; pybind11, like {altar_cuda_buildModule}
  Python_add_library(cudaseismicmodule MODULE WITH_SOABI)
  # adjust the name to match what python expects
  set_target_properties(
    cudaseismicmodule PROPERTIES
    LIBRARY_OUTPUT_NAME cudaseismic
    )
  # set the include directories; altar/ext/cuda provides the shared binding helpers
  target_include_directories(
    cudaseismicmodule PRIVATE
    ${CMAKE_INSTALL_PREFIX}/include
    ${Python3_NumPy_INCLUDE_DIRS}
    ${PYRE_INCLUDE_DIRS}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${CMAKE_CUDA_COMPILER_TOOLKIT_ROOT}/include/cccl
    ${CMAKE_SOURCE_DIR}/altar/ext/cuda
    )
  # set the linker
  set_target_properties(cudaseismicmodule PROPERTIES LINKER_LANGUAGE CUDA)
  # set  the link directories
  target_link_directories(
    cudaseismicmodule PRIVATE
    ${CMAKE_INSTALL_PREFIX}/lib
    )
  # set the libraries to link against
  target_link_libraries(
    cudaseismicmodule PRIVATE
    libcudaseismic pybind11::module cudart cublas ${PYRE_LIBRARIES}
    )
  # add the sources
  target_sources(cudaseismicmodule PRIVATE
    ext/cudaseismic/cudaseismic.cc
    ext/cudaseismic/moment.cc
    ext/cudaseismic/kinematic.cc
    )

  # install the seismic extension
  install(
    TARGETS cudaseismicmodule
    LIBRARY
    DESTINATION ${CMAKE_INSTALL_PREFIX}/packages/altar/models/seismic/ext
    )
endfunction(altar_seismic_cuda_buildModule)

# the scripts
function(altar_seismic_cuda_buildDriver)
  # install the scripts
  install(
    PROGRAMS bin/slipmodel bin/slipmodel.plexus
    DESTINATION bin
    )
  # all done
endfunction(altar_seismic_cuda_buildDriver)

# end of file
