
# build the seas package
function(altar_seas_buildPackage)
  # install the sources straight from the source directory
  install(
    DIRECTORY seas
    DESTINATION ${ALTAR_DEST_PACKAGES}/altar/models
    FILES_MATCHING PATTERN *.py
    )
  # build the package meta-data
  configure_file(
    seas/meta.py.in seas/meta.py
    @ONLY
    )
  # install the generated package meta-data file
  install(
    DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/seas
    DESTINATION ${ALTAR_DEST_PACKAGES}/altar/models
    FILES_MATCHING PATTERN *.py
    )
  # all done
endfunction(altar_seas_buildPackage)


# the scripts
function(altar_seas_buildDriver)
  # install the scripts
  install(
    PROGRAMS bin/SEAS
    DESTINATION bin
    )
  # all done
endfunction(altar_seas_buildDriver)

