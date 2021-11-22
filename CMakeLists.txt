cmake_minimum_required(VERSION 3.13)
project(osqp)

include(CMakePrintHelpers)
set(CMAKE_VERBOSE_MAKEFILE ON)

# Detect operating system
# ----------------------------------------------
message(STATUS "We are on a ${CMAKE_SYSTEM_NAME} system")
if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
  set(IS_LINUX ON)
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
  set(IS_MAC ON)
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
  set(IS_WINDOWS ON)
endif()

# OSQP Versioning
# ----------------------------------------------
set(OSQP_VERSION
    "0.0.0"
    CACHE STRING "The version number of OSQP")
if(NOT OSQP_VERSION STREQUAL "0.0.0")
  configure_file("${PROJECT_SOURCE_DIR}/configure/version.h.in"
                 "${PROJECT_SOURCE_DIR}/include/version.h")
endif()

# Conda environment detection
# ----------------------------------------------
# If MKL_ROOT is not explicitly set, and we're running inside a conda
# environment, set MKL_ROOT to $CONDA_PREFIX to see if we can find MKL (if
# installed through `conda install -c intel mkl-devel`) On a typical OneAPI
# installation outside conda, this would mean specifying:
# -DMKL_DIR=/path/to/oneapi/mkl/latest/lib/cmake/mkl
# -DMKL_ROOT=/path/to/oneapi/mkl/latest
# ----------------------------------------------
if(NOT DEFINED ENV{MKL_ROOT})
  if(DEFINED ENV{CONDA_PREFIX})
    message(
      STATUS "Detected Conda environment - setting MKL_ROOT to $CONDA_PREFIX")
    set(MKL_ROOT $ENV{CONDA_PREFIX})
    include_directories("$ENV{CONDA_PREFIX}/include")
    link_directories("$ENV{CONDA_PREFIX}/lib")
  endif()
endif()

# Export compile commands
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set the output folder where your program will be created
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/out)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/out)

# Prevent generation of output Debug/ folder on MSVC See
# https://stackoverflow.com/questions/543203
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${EXECUTABLE_OUTPUT_PATH}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${EXECUTABLE_OUTPUT_PATH}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG "${LIBRARY_OUTPUT_PATH}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE "${LIBRARY_OUTPUT_PATH}")

# Some non-standard CMake modules
list(APPEND CMAKE_PREFIX_PATH ${PROJECT_SOURCE_DIR}/configure/cmake)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/configure/cmake)
include(FindPythonModule)
include(Utils)

message(STATUS "CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}")

# Set options
# ----------------------------------------------
set(ALGEBRA
    "default"
    CACHE STRING "The Algebra to use (default/mkl/cuda)")

option(PRINTING "Enable solver printing" ON)
option(PROFILING "Enable solver profiling (timing)" ON)
option(CTRLC "Enable user interrupt (Ctrl-C)" ON)

# Is the code generated for embedded platforms? 1 :   Yes. Matrix update not
# allowed. 2 :   No. Matrix update allowed.
if(DEFINED EMBEDDED)
  message(STATUS "Forcing ALGEBRA=default for EMBEDDED mode")
  set(ALGEBRA "default")
  message(STATUS "Disabling printing/profiling/Ctrl-C for EMBEDDED mode")
  set(PRINTING OFF)
  set(PROFILING OFF)
  set(CTRLC OFF)
  message(STATUS "EMBEDDED = ${EMBEDDED}")
else()
  message(STATUS "EMBEDDED = OFF")
endif()

# Display final ALGEBRA chosen and set internal boolean variables
message(STATUS "ALGEBRA = ${ALGEBRA}")
if(${ALGEBRA} STREQUAL "default")
  set(ALGEBRA_DEFAULT ON)
elseif(${ALGEBRA} STREQUAL "mkl")
  set(ALGEBRA_MKL ON)
elseif(${ALGEBRA} STREQUAL "cuda")
  set(ALGEBRA_CUDA ON)
endif()

if(ALGEBRA_CUDA)
  option(DFLOAT "Use floats instead of doubles" ON)
else()
  option(DFLOAT "Use floats instead of doubles" OFF)
endif()

if(DFLOAT AND ALGEBRA_MKL)
  message(
    STATUS
      "Disabling DFLOAT for MKL (Intel RCI ISS does not support single-precision values yet.)"
  )
  set(DFLOAT OFF)
endif()
message(STATUS "DFLOAT = ${DFLOAT}")

option(DLONG "Use long integers (64bit) for indexing" ON)
if(NOT (CMAKE_SIZEOF_VOID_P EQUAL 8))
  message(STATUS "Disabling long integers (64bit) on 32bit machine")
  set(DLONG OFF)
elseif(ALGEBRA_CUDA)
  message(STATUS "Disabling long integers (64bit) for CUDA")
  set(DLONG OFF)
endif()
message(STATUS "DLONG = ${DLONG}")

option(DEBUG "Debug mode" OFF)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(DEBUG ON)
endif()
message(STATUS "DEBUG = ${DEBUG}")

if(DEBUG)
  set(CMAKE_VERBOSE_MAKEFILE ON)
endif()

# Add code coverage
option(COVERAGE "Perform code coverage" OFF)
message(STATUS "COVERAGE = ${COVERAGE}")

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/configure/osqp_configure.h.in
  ${CMAKE_CURRENT_SOURCE_DIR}/include/osqp_configure.h NEWLINE_STYLE LF)

# CUDA support
# ----------------------------------------------
if(ALGEBRA_CUDA)
  enable_language(CUDA)
  find_package(CUDA)

  # cublas and cusparse headers
  include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

  # sm_75 -> Turing support
  if(DFLOAT)
    set(CMAKE_CUDA_ARCHITECTURES 52 75)
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --gpu-architecture=compute_52
    # --gpu-code=sm_52,sm_75")
  else()
    # To use doubles we need compute capability 6.0 for atomic operations
    set(CMAKE_CUDA_ARCHITECTURES 60 75)
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --gpu-architecture=compute_60
    # --gpu-code=sm_60,sm_75")
  endif()
endif()

# Memory allocators
# ----------------------------------------------
# Report on custom user header options. This is intended to allow users to
# provide definitions of their own memory functions The header should define the
# functions as follows
#
# define c_malloc mymalloc define c_calloc mycalloc define c_realloc myrealloc
# define c_free myfree

if(OSQP_CUSTOM_MEMORY)
  message(STATUS "User custom memory management header: ${OSQP_CUSTOM_MEMORY}")
endif()

# Set Compiler flags
# ----------------------------------------------
set(CMAKE_POSITION_INDEPENDENT_CODE ON) # -fPIC

if(NOT MSVC)
  if(COVERAGE)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --coverage")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
    if(FORTRAN)
      set(CMAKE_FORTRAN_FLAGS "${CMAKE_FORTRAN_FLAGS} --coverage")
    endif(FORTRAN)
  endif()

  if(DEBUG)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0 -g")
  else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
  endif()

  set(CMAKE_C_STANDARD_LIBRARIES "${CMAKE_C_STANDARD_LIBRARIES} -lm") # Include
                                                                      # math
  # Include real time library in linux
  if(IS_LINUX)
    set(CMAKE_C_STANDARD_LIBRARIES "${CMAKE_C_STANDARD_LIBRARIES} -lrt -ldl")
  endif()
endif(NOT MSVC)

# ----------------------------------------------
# OSQPLIB
# ----------------------------------------------
add_library(OSQPLIB OBJECT "")

add_subdirectory(include)
add_subdirectory(src)
add_subdirectory(algebra)

get_property(
  osqplib_includes
  TARGET OSQPLIB
  PROPERTY INCLUDE_DIRECTORIES)
get_property(
  osqplib_sources
  TARGET OSQPLIB
  PROPERTY SOURCES)

# ----------------------------------------------
# osqpstatic
# ----------------------------------------------
add_library(osqpstatic STATIC $<TARGET_OBJECTS:OSQPLIB>)
if(ALGEBRA_DEFAULT)
  # Transitive dependency on OBJECT library does not work see
  # https://gitlab.kitware.com/cmake/cmake/-/issues/18682
  target_sources(osqpstatic PRIVATE $<TARGET_OBJECTS:qdldlobject>)
endif()

# ----------------------------------------------
# osqp_demo
# ----------------------------------------------
if(NOT EMBEDDED)
  add_executable(osqp_demo ${PROJECT_SOURCE_DIR}/examples/osqp_demo.c
                           ${osqplib_sources})
  target_include_directories(osqp_demo PRIVATE ${osqplib_includes})
  if(ALGEBRA_MKL)
    target_link_libraries(osqp_demo osqpstatic $<LINK_ONLY:MKL::MKL>)
  elseif(ALGEBRA_CUDA)
    target_link_libraries(osqp_demo osqpstatic cublas cusparse)
  endif()
endif()

# ----------------------------------------------
# Testing
# ----------------------------------------------
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  include(CTest)
endif()

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME
   AND BUILD_TESTING
   AND NOT EMBEDDED)
  add_subdirectory(tests)
endif()
