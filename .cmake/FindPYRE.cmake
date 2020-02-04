# -*- cmake -*-
#
# Lijun Zhu (ljzhu@gps.caltech.edu)
# (c) 2019-2020 all rights reserved

# Find the pyre library
# The pyre (as well as pyrecuda) lib and include paths will be returned
#  PYRE_FOUND        - system has pyre
#  PYRE_CUDA_FOUND    - system has pyre cuda support
#  PYRE_INCLUDE_DIRS -
#  PYRE_JOURNAL_LIBRARY -
#  PYRE_LIBRARY
#  PYRE_CUDA_LIBRARY
#  PYRE_LIBRARIES - the pyre library directory
#  PYRE_CUDA_LIBRARIES

# pyre is discovered by the ``pyre-config --prefix`` command
find_program(
  PYRE_CONFIG
  NAMES pyre-config
  )
if(PYRE_CONFIG)
  # get the pyre prefix
  execute_process(
    COMMAND pyre-config --prefix
    OUTPUT_VARIABLE PYRE_PREFIX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )
  message(STATUS "${PYRE_PREFIX_PATH}")
else()
    message(STATUS "pyre-config is not found")
endif()

# libpyre
find_library(
  PYRE_LIBRARY
  NAMES pyre
  PATHS ${PYRE_PREFIX_PATH}/lib
  )

# libjournal
find_library(
  PYRE_JOURNAL_LIBRARY
  NAMES journal
  PATHS ${PYRE_PREFIX_PATH}/lib
  )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  PYRE DEFAULT_MSG
  PYRE_PREFIX_PATH PYRE_LIBRARY PYRE_JOURNAL_LIBRARY
  )

mark_as_advanced(PYRE_FOUND)
set(PYRE_INCLUDE_DIRS ${PYRE_PREFIX_PATH}/include)
set(PYRE_LIBRARIES ${PYRE_LIBRARY} ${PYRE_JOURNAL_LIBRARY})

# check pyre cuda
# the header file
find_path(
  PYRE_CUDA_INCLUDE_DIR
  NAMES pyre/cuda.h
  PATHS ${PYRE_PREFIX_PATH}/include
  )

# the library libpyrecuda
find_library(
  PYRE_CUDA_LIBRARY
  NAMES pyrecuda
  PATHS ${PYRE_PREFIX_PATH}/lib
  )

# set flags accordingly
if(PYRE_CUDA_INCLUDE_DIR AND PYRE_CUDA_LIBRARY)
  set(PYRE_CUDA_FOUND TRUE)
  set(PYRE_LIBRARIES ${PYRE_LIBRARIES} ${PYRE_CUDA_LIBRARY})
  set(PYRE_CUDA_LIBRARIES ${PYRE_LIBRARIES} ${PYRE_CUDA_LIBRARY})
else()
  set(PYRE_CUDA_FOUND TRUE)
endif()


# end of file
