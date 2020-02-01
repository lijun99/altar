# -*- cmake -*-
#
# Lijun Zhu (ljzhu@gps.caltech.edu)
# (c) 2019-2020 all rights reserved

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
    PROGRAMS bin/seismic bin/H5Converter
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
    ${GSL_INCLUDE_DIRS}
    ${Python3_NumPy_INCLUDE_DIRS}
    ${PYRECUDA_INCLUDE_DIRS}
    )
  # set the link directories
  target_link_directories(
    libcudaseismic PRIVATE
    ${CMAKE_INSTALL_PREFIX}/lib
    )
  # add the dependencies
  target_link_libraries(
    libcudaseismic PRIVATE
    ${GSL_LIBRARIES} journal
    )
  # add the sources
  target_sources(
    libcudaseismic PRIVATE
    lib/libcudaseismic/cudaKinematicG_kernels.cu
    lib/libcudaseismic/cudaKinematicG.cu
    lib/libcudaseismic/version.cc
    )

  # copy the seismic headers; note the trickery with the terminating slash in the source
  # directory that let's us place the files in the correct destination
  file(
    COPY lib/libcudaseismic/
    DESTINATION ${CMAKE_INSTALL_PREFIX}/${ALTAR_DEST_INCLUDE}/altar/models/seismic/cuda
    FILES_MATCHING PATTERN *.h PATTERN *.icc
    )

  # install the library
  install(
    TARGETS libcudaseismic
    LIBRARY DESTINATION lib
    )

  # all done
endfunction(altar_seismic_cuda_buildLibrary)


# build the seismic extension module
function(altar_seismic_cuda_buildModule)
  # seismic
  Python3_add_library(cudaseismicmodule MODULE)
  # adjust the name to match what python expects
  set_target_properties(
    cudaseismicmodule PROPERTIES
    LIBRARY_OUTPUT_NAME cudaseismic
    SUFFIX ${PYTHON3_SUFFIX}
    )
  # set the include directories
  target_include_directories(
    cudaseismicmodule PRIVATE
    ${CMAKE_INSTALL_PREFIX}/include
    ${GSL_INCLUDE_DIRS}
    ${Python3_NumPy_INCLUDE_DIRS}
    ${PYRECUDA_INCLUDE_DIRS}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
  # set the linker
  set_target_properties(cudaseismicmodule PROPERTIES LINKER_LANGUAGE CUDA)
  # set  the link directories
  target_link_directories(
    cudaseismicmodule PRIVATE
    ${CMAKE_INSTALL_PREFIX}/lib
    )
  # set the libraries to link against
  set(CUDA_LIBRARIES cublas cusolver curand ${PYRECUDA_LIBRARIES})
  target_link_libraries(
    cudaseismicmodule PRIVATE
    libcudaseismic libcudaaltar libaltar journal
    ${CUDA_LIBRARIES}
    )
  # add the sources
  target_sources(cudaseismicmodule PRIVATE
    ext/cudaseismic/cudaseismic.cc
    ext/cudaseismic/metadata.cc
    ext/cudaseismic/kinematicg.cc
    )

  # install the capsule
  install(
    FILES ext/cudaseismic/capsules.h
    DESTINATION ${ALTAR_DEST_INCLUDE}/altar/models/seismic/cuda
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
    PROGRAMS bin/slipmodel bin/kinematicForwardModel
    DESTINATION bin
    )
  # all done
endfunction(altar_seismic_cuda_buildDriver)

# end of file
