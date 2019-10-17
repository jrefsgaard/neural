cmake_minimum_required(VERSION 3.8)
project(libNeural VERSION 1.0 LANGUAGES CXX)

# Tell cmake where to look for special cmake modules.
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

#Specify default build type (This is important for optimisation)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

#Make sure the library finds its dependencies, even after installation.
################################
#Make sure to have the full runtime-path to dependencies that may be installed in non-standard folders.
# use, i.e. don't skip the full RPATH for the build tree
SET(CMAKE_SKIP_BUILD_RPATH  FALSE)

# when building, don't use the install RPATH already
# (but later on when installing)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# the RPATH to be used when installing, but only if it's not a system directory
LIST(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
IF("${isSystemDir}" STREQUAL "-1")
   SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
ENDIF("${isSystemDir}" STREQUAL "-1")
################################

# Find files to compile
file(GLOB_RECURSE HEADERS "include/*")
file(GLOB_RECURSE SOURCE    "source/*.cpp")
add_library(Neural SHARED ${HEADERS} ${SOURCE})

target_include_directories(Neural
  PUBLIC 
    $<INSTALL_INTERFACE:include>    
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

#We use modern features.
target_compile_features(Neural PUBLIC cxx_std_11)

#Add test-executable.
add_executable(test main.cpp)
target_link_libraries(test Neural ${LIBRARIES})
set_target_properties(test PROPERTIES OUTPUT_NAME test)

#I modified the FindArmadillo.cmake to provide an imported target Armadillo::Armadillo
find_package (Armadillo REQUIRED)

target_link_libraries(Neural
  PRIVATE
  PUBLIC
    Armadillo::Armadillo   
)

#Install the library and headers
include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/Neural)

install(TARGETS Neural
    EXPORT neural-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

#Export the targets to a script
install(EXPORT neural-targets
  FILE
    NeuralTargets.cmake
  NAMESPACE
    Neural::
  DESTINATION
    ${INSTALL_CONFIGDIR}
)

include(CMakePackageConfigHelpers)
configure_package_config_file(${CMAKE_CURRENT_LIST_DIR}/cmake/NeuralConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/NeuralConfig.cmake
    INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

#Install the config, and custom find modules
install(FILES
    ${CMAKE_CURRENT_LIST_DIR}/cmake/FindNeural.cmake
    ${CMAKE_CURRENT_LIST_DIR}/cmake/FindArmadillo.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/NeuralConfig.cmake
    DESTINATION ${INSTALL_CONFIGDIR}
)

##############################################
## Exporting from the build tree
configure_file(${CMAKE_CURRENT_LIST_DIR}/cmake/FindNeural.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/FindNeural.cmake
COPYONLY)

export(EXPORT neural-targets FILE ${CMAKE_CURRENT_BINARY_DIR}/NeuralTargets.cmake NAMESPACE Neural::)
