# -*- cmake -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# (c) 2003-2020 all rights reserved

# build the altar_cuda package
function(altar_cuda_buildPackage)
  # install the sources straight from the source directory
  install(
    DIRECTORY cuda
    DESTINATION ${ALTAR_DEST_PACKAGES}/altar
    FILES_MATCHING PATTERN *.py
    )
  # all done
endfunction(altar_cuda_buildPackage)


# buld the altar_cuda libraries
function(altar_cuda_buildLibrary)
  # the libcudaaltar target
  add_library(libcudaaltar SHARED)
  # adjust the name
  set_target_properties(
    libcudaaltar PROPERTIES
    LIBRARY_OUTPUT_NAME cudaaltar
    )

  # set the include directories
  target_include_directories(
    libcudaaltar PRIVATE
    ${CMAKE_INSTALL_PREFIX}/include
    ${GSL_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS} ${PYRECUDA_INCLUDE_DIRS}
    )
  # add the dependencies
  target_link_libraries(
    libcudaaltar PRIVATE
    ${GSL_LIBRARIES}
    )
  # add the sources
  target_sources(
    libcudaaltar PRIVATE
    lib/libcudaaltar/bayesian/cudaMetropolis.cu
    lib/libcudaaltar/norm/cudaL2.cu
    lib/libcudaaltar/distributions/cudaTGaussian.cu
    lib/libcudaaltar/distributions/cudaGaussian.cu
    lib/libcudaaltar/distributions/cudaRanged.cu
    lib/libcudaaltar/distributions/cudaUniform.cu
    )

  # copy the altar headers; note the trickery with the terminating slash in the source
  # directory that let's us place the files in the correct destination
  file(
    COPY lib/libcudaaltar/
    DESTINATION ${CMAKE_INSTALL_PREFIX}/${ALTAR_DEST_INCLUDE}/altar/cuda
    FILES_MATCHING PATTERN *.h PATTERN *.icc
    )
  # install the library
  install(
    TARGETS libcudaaltar
    LIBRARY DESTINATION lib
    )
  # all done
endfunction(altar_cuda_buildLibrary)


# build the altar cuda extension module
function(altar_cuda_buildModule)
  # altar
  Python3_add_library(cudaaltarmodule MODULE)
  # adjust the name to match what python expects
  set_target_properties(
    cudaaltarmodule PROPERTIES
    LIBRARY_OUTPUT_NAME cudaaltar
    SUFFIX ${PYTHON3_SUFFIX}
    )
  # set the include directories
  target_include_directories(
    cudaaltarmodule PRIVATE
    ${CMAKE_INSTALL_PREFIX}/include
    ${GSL_INCLUDE_DIRS}
    ${Python3_NumPy_INCLUDE_DIRS}
    ${PYRECUDA_INCLUDE_DIRS}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
  # set the linker
  set_target_properties(cudaaltarmodule PROPERTIES LINKER_LANGUAGE CUDA)
  # set  the link directories
  target_link_directories(
    cudaaltarmodule PRIVATE
    ${CMAKE_INSTALL_PREFIX}/lib
    )
  # set the libraries to link against
  set(CUDA_LIBRARIES cublas cusolver curand ${PYRECUDA_LIBRARIES})
  target_link_libraries(
    cudaaltarmodule PRIVATE
    libcudaaltar libaltar journal
    ${CUDA_LIBRARIES}
    )
  # add the sources
  target_sources(cudaaltarmodule PRIVATE
    ext/cudaaltar.cc
    ext/metadata.cc
    ext/distributions.cc
    ext/metropolis.cc
    ext/norm.cc
    )

  # install the altar extension
  install(
    TARGETS cudaaltarmodule
    LIBRARY
    DESTINATION ${CMAKE_INSTALL_PREFIX}/packages/altar/cuda/ext
    )
endfunction(altar_cuda_buildModule)


# the scripts
function(altar_cuda_buildDriver)
  # install the scripts
  install(
    PROGRAMS bin/cudaaltar
    DESTINATION bin
    )
  # all done
endfunction(altar_cuda_buildDriver)

# end of file
