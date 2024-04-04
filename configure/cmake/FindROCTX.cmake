# Find the ROCTX library
#
# Input to the module:
#  ROCTRACER_ROOT_DIR - The root directory containing ROCTracer
#
# Output of the module:
#  ROCTX_INCLUDE_DIR - include directory for roctx.h
#  ROC::ROCTX - Imported library target for the ROCTX library


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

find_path(ROCTX_INCLUDE_DIR roctx.h
          PATHS
            /usr/include
            /usr/local/include
            /opt/rocm/include
            ${ROCTRACER_ROOT_DIR}/include
          PATH_SUFFIXES
            roctracer)

if(ROCTX_INCLUDE_DIR)
    message(STATUS "Found roctx header ${ROCTX_INCLUDE_DIR}/roctx.h")
else()
    message(FATAL_ERROR "Unable to find roctx.h")
endif()


find_library(roctx_lib
            NAMES
            ${LIB_PREFIX}roctx64${DLL_EXT}
            PATHS
            /usr/lib
            /opt/rocm/lib
            ${ROCTRACER_ROOT_DIR}/lib/)
    
if(roctx_lib)
    message(STATUS "Found roctx shared library ${roctx_lib}")
    add_library(ROC::ROCTX SHARED IMPORTED)
    set_target_properties(ROC::ROCTX PROPERTIES
        IMPORTED_LINK_INTERFACE_LANGUAGES "C"
        IMPORTED_LOCATION ${roctx_lib})
else()
    message(FATAL_ERROR "Unable to locate roctx library")
endif()

target_include_directories(ROC::ROCTX
    INTERFACE
      ${ROCTX_INCLUDE_DIR})
