# -*- cmake -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# build pyre from its submodule, extern/pyre, as part of this project, and set what FindPYRE
# would have found; in a wheel, pyre's packages, libraries and scripts go where altar's do
macro(altar_bundledPyre)
  set(_pyre ${CMAKE_CURRENT_SOURCE_DIR}/extern/pyre)
  if(NOT EXISTS ${_pyre}/CMakeLists.txt)
    message(FATAL_ERROR "extern/pyre is empty; run: git submodule update --init")
  endif()

  # pyre's cuda support follows altar's, if there is a toolkit
  if(WITH_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
      set(WITH_CUDA OFF CACHE BOOL "enable support for CUDA" FORCE)
    endif()
  endif()
  # not pyre's tests
  set(PYRE_BUILD_TESTING OFF CACHE BOOL "Build pyre's test suite" FORCE)

  if(SKBUILD)
    # pyre's packages at the root of the wheel, its libraries with altar's, its scripts on the path
    set(PYRE_DEST_PACKAGES .)
    set(CMAKE_INSTALL_LIBDIR altar.libs CACHE PATH "the shared libraries, in the wheel" FORCE)
    set(CMAKE_INSTALL_BINDIR ${SKBUILD_SCRIPTS_DIR} CACHE PATH "the scripts, in the wheel" FORCE)
    # the libraries and extension modules find altar.libs relative to themselves, from any depth
    # of the packages, so the installed wheel works wherever pip puts it
    set(CMAKE_INSTALL_RPATH "$ORIGIN")
    foreach(_up .. ../.. ../../.. ../../../.. ../../../../..)
      list(APPEND CMAKE_INSTALL_RPATH "$ORIGIN/${_up}/altar.libs")
    endforeach()
    # and the libraries from outside the wheel, e.g. the cuda toolkit's, where they were found
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
  endif()

  add_subdirectory(${_pyre} ${CMAKE_CURRENT_BINARY_DIR}/extern/pyre)

  if(SKBUILD)
    # pyre sets an absolute rpath of its own; its targets get the relative one
    altar_relocatableRpath(${_pyre})
  endif()

  # what FindPYRE would have found
  set(PYRE_FOUND TRUE)
  set(PYRE_CUDA_FOUND ${WITH_CUDA})
  set(PYRE_LIBRARY pyre::pyre)
  set(PYRE_JOURNAL_LIBRARY pyre::journal)
  set(PYRE_LIBRARIES pyre::pyre pyre::journal)
  set(PYRE_CUDA_LIBRARIES ${PYRE_LIBRARIES})
  set(PYRE_INCLUDE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/extern/pyre/lib)
  set(PYRE_PREFIX_PATH ${CMAKE_INSTALL_PREFIX})
endmacro(altar_bundledPyre)


# give the libraries, modules and executables of {directory} and below the install rpath
function(altar_relocatableRpath directory)
  get_property(targets DIRECTORY ${directory} PROPERTY BUILDSYSTEM_TARGETS)
  foreach(target ${targets})
    get_target_property(type ${target} TYPE)
    if(type MATCHES "SHARED_LIBRARY|MODULE_LIBRARY|EXECUTABLE")
      set_target_properties(${target} PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_RPATH}")
    endif()
  endforeach()
  get_property(subdirectories DIRECTORY ${directory} PROPERTY SUBDIRECTORIES)
  foreach(subdirectory ${subdirectories})
    altar_relocatableRpath(${subdirectory})
  endforeach()
endfunction(altar_relocatableRpath)


# end of file
