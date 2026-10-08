# -*- cmake -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# (c) 2003-present all rights reserved

#
# Example build steps:
#     mkdir build && cd build
#     cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DCMAKE_PREFIX_PATH=$CONDA_PREFIX -DCMAKE_CUDA_ARCHITECTURES=native
#     make -j && make install

# cmake setup
# v3.18 is needed to pass Development.Module to FIndPython
cmake_minimum_required(VERSION 3.19...3.25)

# options
option(WITH_CUDA "enable support for CUDA" ON)

# adjust the include path
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/.cmake)
# get support
include(altar_init)

# ask git for the altar version
altar_getVersion()

# set up the project
project(ALTAR VERSION ${REPO_MAJOR}.${REPO_MINOR}.${REPO_MICRO} LANGUAGES CXX)

# packages
# mpi
find_package(MPI)

# python
set(PYTHON_COMPONENTS Interpreter Development.Module NumPy)
find_package(Python 3.7 REQUIRED COMPONENTS ${PYTHON_COMPONENTS} OPTIONAL_COMPONENTS Development.Embed)

# pyre and pyre cuda
find_package(PYRE)

# pybind11, for the python extension modules
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11)

# set up cmake
altar_cmakeInit()
# set up c++
altar_cxxInit()
# set up python
altar_pythonInit()
# set up cuda
if (WITH_CUDA AND PYRE_CUDA_FOUND)
  # check whether cuda toolkit is available
  include(CheckLanguage)
  check_language(CUDA)
  # available
  if(CMAKE_CUDA_COMPILER)
    # turn it on
    enable_language(CUDA)
    message(STATUS "CUDA Toolkit found and CUDA support is enabled")
    # other cuda settings
    # enforce linker for CUDA modules; not properly set for some systems
    set(CMAKE_CUDA_HOST_LINK_LAUNCHER ${CMAKE_CXX_COMPILER})
    if(NOT DEFINED CMAKE_CUDA_STANDARD)
      # pyre's own headers are c++23 now (e.g. {std::endian} in lib/pyre/memory/Ordered.h),
      # and any .cu file that reaches them needs the same standard nvcc's own side compiles
      # against; c++11 predates that entirely
      set(CMAKE_CUDA_STANDARD 20)
      set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    endif()
  # cuda toolkit not available
  else()
    # turn off WITH_CUDA option
    message(STATUS "CUDA Toolkit or Pyre cuda extension not found; set WITH_CUDA to OFF")
    set(WITH_CUDA OFF)
  endif()
endif()

# initialize the variables that describe the staging directory layout
altar_stagingInit()
# initialize the variables that describe the install directory layout
altar_destinationInit()

# build the framework
add_subdirectory(altar)

if(WITH_CUDA)
  include(altar_cudaframework)
  altar_cuda_buildPackage()
  altar_cuda_buildLibrary()
  altar_cuda_buildModule()
endif()

# build the models
add_subdirectory(models/emhp)
add_subdirectory(models/gaussian)
add_subdirectory(models/mogi)
add_subdirectory(models/cdm)
add_subdirectory(models/reverso)
add_subdirectory(models/linear)
add_subdirectory(models/seismic)
add_subdirectory(models/seas)
add_subdirectory(models/regression)

# end of file
