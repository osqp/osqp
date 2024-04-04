# Find the ITTAPI library
#
# Input to the module:
#  ITTAPI_ARCH - Architecture to use (32 vs 64)
#  ITTAPI_LINK - The link to use (shared vs static)
#  ITTAPI_ROOT_DIR - The root directory containing ITT API
#
# This will also look at the environment variable VTUNE_PROFILER_DIR for ITT.
#
# Output of the module:
#  ITTNOTIFY_INCLUDE_DIR - include directory for ittnotify.h
#  ITTAPI::ittnotify - Imported library target for the ITT library
#

# Path from VTune profiler installation
if(DEFINED ENV{VTUNE_PROFILER_DIR})
    set(VTUNE_PROFILER_DIR $ENV{VTUNE_PROFILER_DIR})
    message(STATUS "VTune installation directory ${VTUNE_PROFILER_DIR}")
endif()

# Extensions
if(UNIX)
    set(LIB_PREFIX "lib")
    set(LIB_EXT ".a")
    set(DLL_EXT ".so")
    if(APPLE)
        set(DLL_EXT ".dylib")
    endif()
    set(LINK_PREFIX "-l")
    set(LINK_SUFFIX "")
else()
    set(LIB_PREFIX "")
    set(LIB_EXT ".lib")
    set(DLL_EXT "_dll.lib")
    set(LINK_PREFIX "")
    set(LINK_SUFFIX ".lib")
endif()

find_path(ITTNOTIFY_INCLUDE_DIR ittnotify.h
          PATHS
            /usr/include
            /usr/local/include
            ${VTUNE_PROFILER_DIR}/sdk/include       # Path from VTune installation
            ${ITTAPI_ROOT_DIR}
          PATH_SUFFIXES
            ittapi
            ittnotify)

if(ITTNOTIFY_INCLUDE_DIR)
    message(STATUS "Found ittnotify header ${ITTNOTIFY_INCLUDE_DIR}/ittnotify.h")
else()
    message(FATAL_ERROR "Unable to find ittnotify.h")
endif()

find_library(ittnotify_lib
            NAMES
              ${LIB_PREFIX}ittnotify${LIB_EXT}
              ittnotify
            PATHS
              /usr/lib
              ${VTUNE_PROFILER_DIR}/sdk/
              ${ITTAPI_ROOT_DIR}
            PATH_SUFFIXES
              "lib"
              "lib${ITTAPI_ARCH}")

if(ittnotify_lib)
    message(STATUS "Found ittnotify static library ${ittnotify_lib}")
    add_library(ITTAPI::ittnotify STATIC IMPORTED)
else()
    find_library(ittnotify_lib
                NAMES
                  ${LIB_PREFIX}ittnotify${DLL_EXT}
                  ittnotify
                PATHS
                  /usr/lib
                  ${VTUNE_PROFILER_DIR}/sdk/
                PATH_SUFFIXES
                  "lib"
                  "lib/${ITTAPI_ARCH}")
    
    if(ittnotify_lib)
        message(STATUS "Found ittnotify shared library ${ittnotify_lib}")
        add_library(ITTAPI::ittnotify SHARED IMPORTED)
    else()
        message(FATAL_ERROR "Unable to locate ittnotify library")
    endif()
endif()

set_target_properties(ITTAPI::ittnotify PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES "C"
    IMPORTED_LOCATION ${ittnotify_lib})

target_link_libraries(ITTAPI::ittnotify
    INTERFACE
      pthread
      dl)

target_include_directories(ITTAPI::ittnotify
    INTERFACE
      ${ITTNOTIFY_INCLUDE_DIR})
