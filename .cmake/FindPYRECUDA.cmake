# -*- cmake -*-
#
# Lijun Zhu (ljzhu@gps.caltech.edu)
# (c) 2019-2020 all rights reserved

# Find the pyre cuda library
# pyre search path can be passed by
#   1) the $LD_LIBRARY_PATH env variable
#   2) cmake -DPYRE_ROOT_DIR=... or
#   3) cmake -DCMAKE_PREFIX_PATH=...
# The pyrecuda (as well as pyre) lib and include paths will be returned
#  PYRECUDA_FOUND        - system has pyre
#  PYRECUDA_INCLUDE_DIRS - the pyre include directory
#  PYRECUDA_LIBRARIES - the pyre library directory

# add LD_LIBRARY_PATH to search path
string(REPLACE ":" ";" LIBRARY_DIRS $ENV{LD_LIBRARY_PATH})

find_library(
  PYRECUDA_LIBRARY
  NAMES pyrecuda
  PATHS ${LIBRARY_DIRS} ${PYRE_ROOT_DIR}/lib
  )

# get the library path for searching the include dir
get_filename_component(
  PYRECUDA_LIBRARY_PATH
  ${PYRECUDA_LIBRARY} DIRECTORY
  )

find_path(
  PYRECUDA_INCLUDE_DIR
  NAMES pyre/cuda.h
  PATHS ${PYRE_ROOT_DIR}/include ${PYRECUDA_LIBRARY_PATH}/../include
  )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  PYRECUDA DEFAULT_MSG
  PYRECUDA_LIBRARY PYRECUDA_INCLUDE_DIR
  )

mark_as_advanced(PYRECUDA_FOUND)
set(PYRECUDA_INCLUDE_DIRS ${PYRECUDA_INCLUDE_DIR})
set(PYRECUDA_LIBRARIES ${PYRECUDA_LIBRARY})


# end of file
