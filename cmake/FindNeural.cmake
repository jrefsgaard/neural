# FindNeural.cmake
#
# Finds the Neural library
#
# This will define the following variables
#
#    Neural_FOUND
#    Neural_INCLUDE_DIRS
#    Neural_LIBRARIES
#
# and the following imported targets
#
#     Neural::Neural  
#
# Author: Jonas Refsgaard (jrefsgaard@triumf.ca)

find_package(PkgConfig)
pkg_check_modules(PC_Neural QUIET Neural)

find_path(Neural_INCLUDE_DIR
  NAMES neural.h
  PATHS ${PC_Neural_INCLUDE_DIRS}
  PATH_SUFFIXES neural
)
find_library(Neural_LIBRARY
  NAMES Neural
  PATHS ${PC_Neural_LIBRARY_DIRS}
)

set(Nerual_VERSION ${PC_Neural_VERSION})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Neural
  FOUND_VAR Neural_FOUND
  REQUIRED_VARS
    Neural_LIBRARY
    Neural_INCLUDE_DIR
  VERSION_VAR Neural_VERSION
)

#The old-school way
if(Neural_FOUND)
  set(Neural_LIBRARIES ${Neural_LIBRARY})
  set(Neural_INCLUDE_DIRS ${Neural_INCLUDE_DIR})
  set(Neural_DEFINITIONS ${PC_Neural_CFLAGS_OTHER})
endif()

#The modern way
if(Neural_FOUND AND NOT TARGET Neural::Neural)
  add_library(Neural::Neural UNKNOWN IMPORTED)
  set_target_properties(Neural::Neural PROPERTIES
    IMPORTED_LOCATION "${Neural_LIBRARY}"
    INTERFACE_COMPILE_OPTIONS "${PC_Neural_CFLAGS_OTHER}"
    INTERFACE_INCLUDE_DIRECTORIES "${Neural_INCLUDE_DIR}"
  )
endif()

mark_as_advanced(
  Neural_INCLUDE_DIR
  Neural_LIBRARY
)

