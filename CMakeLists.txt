# -*- cmake -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# (c) 2003-2020 all rights reserved

# cmake setup
cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
# policies
if (POLICY CMP0048)
  cmake_policy(SET CMP0048 NEW)
endif(POLICY CMP0048)
cmake_policy(SET CMP0060 NEW)
if (POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif(POLICY CMP0074)
if (POLICY CMP0076)
  cmake_policy(SET CMP0076 NEW)
endif(POLICY CMP0076)

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
# gsl
find_package(GSL)
# mpi
find_package(MPI)
# python
set(PYTHON_COMPONENTS Interpreter Development)
if(GSL_FOUND)
    list(APPEND PYTHON_COMPONENTS NumPy)
endif()
find_package(Python3 REQUIRED COMPONENTS ${PYTHON_COMPONENTS})
# pyre and pyre cuda
find_package(PYRE)

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
      set(CMAKE_CUDA_STANDARD 11)
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
  add_subdirectory(cuda)
endif()

# build the models
add_subdirectory(models/emhp)
add_subdirectory(models/gaussian)
add_subdirectory(models/mogi)
add_subdirectory(models/cdm)
add_subdirectory(models/linear)
add_subdirectory(models/seismic)
add_subdirectory(models/cudalinear)
add_subdirectory(models/sir)
add_subdirectory(models/regression)

# end of file
