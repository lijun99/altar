# -*- cmake -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# (c) 2003-2021 all rights reserved

# build the altar_cuda package
function(altar_cuda_buildPackage)
  # install the sources straight from the source directory
  install(
    DIRECTORY ${CMAKE_SOURCE_DIR}/altar/cuda
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
    ${GSL_INCLUDE_DIRS}
    ${Python3_NumPy_INCLUDE_DIRS}
    ${PYRE_INCLUDE_DIRS}
    )
  # add the dependencies
  target_link_libraries(
    libcudaaltar PRIVATE
    ${GSL_LIBRARIES}
    ${PYRE_LIBRARIES}
    )
  # {WITH_CUDA} turns on pyre's {PYRE_HOST_DEVICE} decorations on {pyre::grid}/{pyre::memory},
  # needed by any kernel that indexes one of pyre's own grids directly (see cudaL2.cu);
  # {--expt-relaxed-constexpr} lets device code reach the constexpr machinery under pyre's
  # {Shape}/{Index}. Harmless for the other .cu files in this target that don't use pyre.grid
  target_compile_definitions(libcudaaltar PRIVATE WITH_CUDA)
  target_compile_options(libcudaaltar PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  # add the sources
  target_sources(
    libcudaaltar PRIVATE
    ${CMAKE_SOURCE_DIR}/altar/lib/libcudaaltar/bayesian/cudaMetropolis.cu
    ${CMAKE_SOURCE_DIR}/altar/lib/libcudaaltar/bayesian/cudaLeapfrog.cu
    ${CMAKE_SOURCE_DIR}/altar/lib/libcudaaltar/bayesian/cudaLangevin.cu
    ${CMAKE_SOURCE_DIR}/altar/lib/libcudaaltar/norm/cudaL2.cu
    ${CMAKE_SOURCE_DIR}/altar/lib/libcudaaltar/distributions/cudaTGaussian.cu
    ${CMAKE_SOURCE_DIR}/altar/lib/libcudaaltar/distributions/cudaGaussian.cu
    ${CMAKE_SOURCE_DIR}/altar/lib/libcudaaltar/distributions/cudaRanged.cu
    ${CMAKE_SOURCE_DIR}/altar/lib/libcudaaltar/distributions/cudaUniform.cu
    ${CMAKE_SOURCE_DIR}/altar/lib/libcudaaltar/distributions/cudaUniformLogit.cu
    ${CMAKE_SOURCE_DIR}/altar/lib/libcudaaltar/distributions/cudaLogistic.cu
    ${CMAKE_SOURCE_DIR}/altar/lib/libcudaaltar/distributions/cudaTGaussianLogit.cu
    )

  # copy the altar headers; note the trickery with the terminating slash in the source
  # directory that let's us place the files in the correct destination
  # Find all .h and .icc files recursively
  file(GLOB_RECURSE CUDA_ALTAR_HEADERS
    "${CMAKE_SOURCE_DIR}/altar/lib/libcudaaltar/*.h"
    "${CMAKE_SOURCE_DIR}/altar/lib/libcudaaltar/*.icc"
  )

  # If no files are found, trigger a fatal error
  if(NOT CUDA_ALTAR_HEADERS)
    message(FATAL_ERROR "No CUDA altar header files found in ${CMAKE_SOURCE_DIR}/altar/lib/libcudaaltar/")
  endif()

  # Ensure the destination directory exists
  file(MAKE_DIRECTORY ${CMAKE_INSTALL_PREFIX}/${ALTAR_DEST_INCLUDE}/altar/cuda)

  # Custom target to copy each file while preserving directory structure
  set(CUDA_ALTAR_HEADER_OUTPUTS)

  foreach(FILE ${CUDA_ALTAR_HEADERS})
    # Get the relative path of the file inside libcudaaltar
    file(RELATIVE_PATH REL_PATH "${CMAKE_SOURCE_DIR}/altar/lib/libcudaaltar" "${FILE}")

    # Compute the full destination path (preserving structure)
    set(DEST_PATH "${CMAKE_INSTALL_PREFIX}/${ALTAR_DEST_INCLUDE}/altar/cuda/${REL_PATH}")

    # Ensure the destination directory exists
    get_filename_component(DEST_DIR "${DEST_PATH}" DIRECTORY)

    # Add a command to copy the file
    add_custom_command(
        OUTPUT "${DEST_PATH}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${DEST_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${FILE}" "${DEST_PATH}"
        DEPENDS "${FILE}"
    )
    list(APPEND CUDA_ALTAR_HEADER_OUTPUTS "${DEST_PATH}")
  endforeach()

  add_custom_target(copy_cuda_headers ALL DEPENDS ${CUDA_ALTAR_HEADER_OUTPUTS})

  # Install headers while preserving directory structure
  install(DIRECTORY "${CMAKE_SOURCE_DIR}/altar/lib/libcudaaltar/"
    DESTINATION "${CMAKE_INSTALL_PREFIX}/${ALTAR_DEST_INCLUDE}/altar/cuda"
    FILES_MATCHING PATTERN "*.h" PATTERN "*.icc"
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
  # altar; pybind11 now, not the raw cpython api the old capsule-based bindings used
  Python_add_library(cudaaltarmodule MODULE WITH_SOABI)
  # adjust the name to match what python expects
  set_target_properties(
    cudaaltarmodule PROPERTIES
    LIBRARY_OUTPUT_NAME cudaaltar
    )
  # set the include directories
  target_include_directories(
    cudaaltarmodule PRIVATE
    ${CMAKE_INSTALL_PREFIX}/include
    ${GSL_INCLUDE_DIRS}
    ${Python3_NumPy_INCLUDE_DIRS}
    ${PYRE_INCLUDE_DIRS}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    # {CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} doesn't reliably carry the toolkit's bundled
    # CCCL (libcu++) headers -- e.g. <cuda/std/mdspan> -- across cmake versions, so add it
    # explicitly; these .cc files are compiled by the host compiler, not nvcc, so they don't
    # get nvcc's own implicit "-isystem .../include/cccl" the way .cu files do
    ${CMAKE_CUDA_COMPILER_TOOLKIT_ROOT}/include/cccl
    )
  # set the linker
  set_target_properties(cudaaltarmodule PROPERTIES LINKER_LANGUAGE CUDA)
  # set  the link directories
  target_link_directories(
    cudaaltarmodule PRIVATE
    ${CMAKE_INSTALL_PREFIX}/lib
    )
  # set the libraries to link against
  set(CUDA_LIBRARIES cudart cublas cusolver curand ${PYRE_LIBRARIES})
  target_link_libraries(
    cudaaltarmodule PRIVATE
    libcudaaltar libaltar pybind11::module
    ${CUDA_LIBRARIES}
    )
  # add the sources; distributions.cc/metropolis.cc/langevin.cc/leapfrog.cc are not yet
  # ported to pybind11+AnyGrid (see the note in cudaaltar.cc) and are left out of the build
  # until they are
  target_sources(cudaaltarmodule PRIVATE
    ${CMAKE_SOURCE_DIR}/altar/ext/cuda/cudaaltar.cc
    ${CMAKE_SOURCE_DIR}/altar/ext/cuda/metadata.cc
    ${CMAKE_SOURCE_DIR}/altar/ext/cuda/norm.cc
    ${CMAKE_SOURCE_DIR}/altar/ext/cuda/distributions.cc
    ${CMAKE_SOURCE_DIR}/altar/ext/cuda/gaussian.cc
    ${CMAKE_SOURCE_DIR}/altar/ext/cuda/uniform.cc
    ${CMAKE_SOURCE_DIR}/altar/ext/cuda/ranged.cc
    ${CMAKE_SOURCE_DIR}/altar/ext/cuda/tgaussian.cc
    ${CMAKE_SOURCE_DIR}/altar/ext/cuda/logistic.cc
    ${CMAKE_SOURCE_DIR}/altar/ext/cuda/metropolis.cc
    ${CMAKE_SOURCE_DIR}/altar/ext/cuda/leapfrog.cc
    ${CMAKE_SOURCE_DIR}/altar/ext/cuda/langevin.cc
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
