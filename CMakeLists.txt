cmake_minimum_required(VERSION 3.18)
project(osqp)

include(GNUInstallDirs)
include(CMakeDependentOption)
include(CMakePackageConfigHelpers)

# Detect the CMake build type and use that as the debug indicator
if(DEFINED CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "Set to either \"Release\" or \"Debug\"")
else()
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Set to either \"Release\" or \"Debug\"")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(OSQP_ENABLE_DEBUG ON)
  set(CMAKE_VERBOSE_MAKEFILE ON)
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

message(STATUS "Debug mode: ${OSQP_ENABLE_DEBUG}")

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
  configure_file("${PROJECT_SOURCE_DIR}/configure/version.h.in" "${PROJECT_SOURCE_DIR}/include/private/version.h")
endif()

# Conda environment detection
# ----------------------------------------------
# If MKL_ROOT is not explicitly set, and we're running inside a conda environment, set MKL_ROOT to $CONDA_PREFIX to see
# if we can find MKL (if installed through `conda install -c intel mkl-devel`) On a typical OneAPI installation outside
# conda, this would mean specifying: -DMKL_DIR=/path/to/oneapi/mkl/latest/lib/cmake/mkl
# -DMKL_ROOT=/path/to/oneapi/mkl/latest
# ----------------------------------------------
if(NOT DEFINED ENV{MKL_ROOT} AND NOT MKL_ROOT)
  if(DEFINED ENV{CONDA_PREFIX})
    message(STATUS "Detected Conda environment - setting MKL_ROOT to $CONDA_PREFIX")
    set(MKL_ROOT $ENV{CONDA_PREFIX})
    include_directories("$ENV{CONDA_PREFIX}/include")
    link_directories("$ENV{CONDA_PREFIX}/lib")
  endif()
endif()

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/out)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/out)

# Prevent generation of output Debug/ folder on MSVC See https://stackoverflow.com/questions/543203
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${EXECUTABLE_OUTPUT_PATH}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${EXECUTABLE_OUTPUT_PATH}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG "${LIBRARY_OUTPUT_PATH}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE "${LIBRARY_OUTPUT_PATH}")

# Some non-standard CMake modules
list(APPEND CMAKE_PREFIX_PATH ${PROJECT_SOURCE_DIR}/configure/cmake)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/configure/cmake)
include(Utils)

# Set options
# ----------------------------------------------
option(OSQP_BUILD_SHARED_LIB "Build the shared library" ON)
option(OSQP_BUILD_STATIC_LIB "Build the static library" ON)

cmake_dependent_option( OSQP_BUILD_DEMO_EXE
                        "Build the demo executable (requires the static library)"
                        ON    # Default to on
                        OSQP_BUILD_STATIC_LIB OFF ) # Force off if the static library isn't built

cmake_dependent_option( OSQP_BUILD_UNITTESTS
                        "Build the unit testing suite"
                        OFF    # Default to off
                        OSQP_BUILD_STATIC_LIB OFF ) # Force off if the static library isn't built

cmake_dependent_option( OSQP_COVERAGE_CHECK
                        "Check the code coverage of the unit tests"
                        OFF    # Default to off
                        OSQP_BUILD_UNITTESTS OFF ) # Force off if the unit tests aren't built

set(OSQP_ALGEBRA_BACKEND
    "builtin"
    CACHE STRING "The Algebra to use (builtin/mkl/cuda)")

option(OSQP_ENABLE_PRINTING "Enable solver printing" ON)
option(OSQP_ENABLE_PROFILING "Enable solver profiling (timing)" ON)
option(OSQP_ENABLE_INTERRUPT "Enable user interrupt (e.g. Ctrl-C)" ON)

# Allow appending a string to the end of the library and the soname so people can have
# multiple libraries side-by-side on an install.
set(OSQP_LIB_SUFFIX "" CACHE STRING "String to append to the library name")
mark_as_advanced(OSQP_LIB_SUFFIX)

# Set the relevant boolean values for the algebra
if(${OSQP_ALGEBRA_BACKEND} STREQUAL "builtin")
  set(OSQP_ALGEBRA_BUILTIN ON)
elseif(${OSQP_ALGEBRA_BACKEND} STREQUAL "mkl")
  set(OSQP_ALGEBRA_MKL ON)
elseif(${OSQP_ALGEBRA_BACKEND} STREQUAL "cuda")
  set(OSQP_ALGEBRA_CUDA ON)
endif()

# Is the code generated for embedded platforms? 1 :   Yes. Matrix update not allowed. 2 :   No. Matrix update allowed.
if(DEFINED OSQP_EMBEDDED_MODE)
  if(NOT OSQP_ALGEBRA_BUILTIN)
    message(WARNING "Forcing OSQP_ALGEBRA_BACKEND=builtin for OSQP_EMBEDDED_MODE mode.")
  endif()

  set(OSQP_ALGEBRA_BACKEND "builtin")
  set(OSQP_ALGEBRA_BUILTIN ON)
  unset(OSQP_ALGEBRA_CUDA)
  unset(OSQP_ALGEBRA_MKL)

  if(OSQP_ENABLE_PRINTING AND NOT OSQP_CUSTOM_PRINTING)
    message(WARNING "Disabling printing in OSQP_EMBEDDED_MODE mode.")
    set(OSQP_ENABLE_PRINTING OFF)
  endif()

  if(OSQP_ENABLE_INTERRUPT AND NOT OSQP_CUSTOM_INTERRUPT)
    message(WARNING "Disabling interrupts in OSQP_EMBEDDED_MODE mode.")
    set(OSQP_ENABLE_INTERRUPT OFF)
  endif()

  if(OSQP_ENABLE_PROFILING AND NOT OSQP_CUSTOM_TIMING)
    message(WARNING "Disabling profiling (timing) for OSQP_EMBEDDED_MODE mode.")
    set(OSQP_ENABLE_PROFILING OFF)
  endif()

  # Disable shared library and demo exe on embedded applications
  if(${OSQP_BUILD_SHARED_LIB} OR ${OSQP_BUILD_DEMO_EXE})
    message(WARNING "Disabling shared library and demo executable for OSQP_EMBEDDED_MODE mode.")
  endif()

  set(OSQP_BUILD_SHARED_LIB OFF)
  set(OSQP_BUILD_DEMO_EXE OFF)

  if(OSQP_EMBEDDED_MODE EQUAL 1)
    message(STATUS "Embedded mode: Vector updates")
  elseif(OSQP_EMBEDDED_MODE EQUAL 2)
    message(STATUS "Embedded mode: Matrix updates")
  else()
    error("Unknown OSQP_EMBEDDED_MODE mode: ${OSQP_EMBEDDED_MODE}")
  endif()
else()
  message(STATUS "Embedded mode: OFF")
endif()

# Display final algebra chosen
message(STATUS "Algebra backend: ${OSQP_ALGEBRA_BACKEND}")

# Display final profiling behaviour
message(STATUS "Solver profiling: ${OSQP_ENABLE_PROFILING}")

# Display final interrupt behaviour
message(STATUS "Solver interrupt: ${OSQP_ENABLE_INTERRUPT}")

if(OSQP_ALGEBRA_CUDA)
  # Some options have different defaults for the CUDA algebra
  option(OSQP_USE_FLOAT "Use floats instead of doubles" ON)
  option(OSQP_USE_LONG "Use long integers (64bit) for indexing" OFF)
else()
  option(OSQP_USE_FLOAT "Use floats instead of doubles" OFF)
  option(OSQP_USE_LONG "Use long integers (64bit) for indexing" ON)
endif()

if(OSQP_USE_FLOAT AND OSQP_ALGEBRA_MKL)
  message(WARNING "Disabling OSQP_USE_FLOAT for MKL (Intel RCI ISS does not support single-precision values yet.)")
  set(OSQP_USE_FLOAT OFF)
endif()

if(OSQP_USE_FLOAT)
  message(STATUS "Using single precision floating-point")
else()
  message(STATUS "Using double precision floating-point")
endif()

if(NOT (CMAKE_SIZEOF_VOID_P EQUAL 8))
  message(WARNING "Disabling long integers (64bit) on 32bit machine.")
  set(OSQP_USE_LONG OFF)
endif()

if(OSQP_ALGEBRA_CUDA AND OSQP_USE_LONG)
  message(WARNING "Disabling long integers (64bit) for CUDA.")
  set(OSQP_USE_LONG OFF)
elseif(DEFINED OSQP_EMBEDDED_MODE)
  # The long long type is not in the C89 spec, and our embedded code must be C89 compliant
  message(STATUS "Disabling long integers (64bit) for embedded mode")
  set(OSQP_USE_LONG OFF)
endif()

if(OSQP_USE_LONG)
  message(STATUS "Using long integers")
else()
  message(STATUS "Using standard (int) integers")
endif()

option(OSQP_ASAN "Enable ASAN" OFF)

cmake_dependent_option( OSQP_CODEGEN "Enable code generation"
                        ON
                        "OSQP_ALGEBRA_BUILTIN;NOT DEFINED OSQP_EMBEDDED_MODE" OFF )

message(STATUS "Code generation: ${OSQP_CODEGEN}")

cmake_dependent_option( OSQP_ENABLE_DERIVATIVES "Enable derivative computation"
                        ON
                        "OSQP_ALGEBRA_BUILTIN;NOT DEFINED EMBEDDED" OFF )

message( STATUS "Derivative support: ${OSQP_ENABLE_DERIVATIVES}" )

# Rename compile-time constants & configure
# ----------------------------------------------
# If we are creating any OSQP_* compile-time constants from CMake variables, do so here.
if(OSQP_ENABLE_PRINTING AND OSQP_CUSTOM_PRINTING)
  message(STATUS "Using custom printing header: ${OSQP_CUSTOM_PRINTING}")
endif()

if(OSQP_CUSTOM_MEMORY)
  message(STATUS "Using custom memory management header: ${OSQP_CUSTOM_MEMORY}")
endif()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/configure/osqp_configure.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/include/public/osqp_configure.h NEWLINE_STYLE LF)


# CUDA support
# ----------------------------------------------
if(OSQP_ALGEBRA_CUDA)
  enable_language(CUDA)
  find_package(CUDA)

  # cublas and cusparse headers
  include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

  # sm_75 -> Turing support
  if(OSQP_USE_FLOAT)
    set(CMAKE_CUDA_ARCHITECTURES 52 75)
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --gpu-architecture=compute_52 --gpu-code=sm_52,sm_75")
  else()
    # To use doubles we need compute capability 6.0 for atomic operations
    set(CMAKE_CUDA_ARCHITECTURES 60 75)
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --gpu-architecture=compute_60 --gpu-code=sm_60,sm_75")
  endif()
endif()


# Set Compiler flags
# ----------------------------------------------
set(CMAKE_POSITION_INDEPENDENT_CODE ON) # -fPIC

if(NOT MSVC)
  if(OSQP_COVERAGE_CHECK)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --coverage")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
    if(FORTRAN)
      set(CMAKE_FORTRAN_FLAGS "${CMAKE_FORTRAN_FLAGS} --coverage")
    endif(FORTRAN)
  endif()

  if(OSQP_ASAN)
        set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -fsanitize=address -fno-optimize-sibling-calls -fsanitize-address-use-after-scope -fno-omit-frame-pointer" )
        set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -D_GLIBCXX_SANITIZE_VECTOR -fsanitize=address -fno-optimize-sibling-calls -fsanitize-address-use-after-scope -fno-omit-frame-pointer" )

        # ASAN shouldn't be used with these options (https://github.com/google/sanitizers/wiki/AddressSanitizer#faq)
        set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-stack-protector -U_FORTIFY_SOURCE" )
        set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-stack-protector -U_FORTIFY_SOURCE" )
  endif()

  if(OSQP_ENABLE_DEBUG)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0 -g")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g")
  else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
  endif()

  set(CMAKE_C_STANDARD_LIBRARIES "${CMAKE_C_STANDARD_LIBRARIES} -lm") # Include math
  # Include real time library in linux
  if(IS_LINUX)
    set(CMAKE_C_STANDARD_LIBRARIES "${CMAKE_C_STANDARD_LIBRARIES} -lrt -ldl")
  endif()
endif(NOT MSVC)

# ----------------------------------------------
# Initialize the code generation file copying
# ----------------------------------------------
if( OSQP_CODEGEN )
  set( EMBEDDED_BUILD_ROOT_DIR "${CMAKE_BINARY_DIR}/codegen_src" )

  set( EMBEDDED_BUILD_SRC_DIR "${EMBEDDED_BUILD_ROOT_DIR}/src" )
  set( EMBEDDED_BUILD_PUBLIC_INC_DIR "${EMBEDDED_BUILD_ROOT_DIR}/inc/public" )
  set( EMBEDDED_BUILD_PRIVATE_INC_DIR "${EMBEDDED_BUILD_ROOT_DIR}/inc/private" )

  file( MAKE_DIRECTORY ${EMBEDDED_BUILD_ROOT_DIR} )
  file( MAKE_DIRECTORY ${EMBEDDED_BUILD_SRC_DIR} )
  file( MAKE_DIRECTORY ${EMBEDDED_BUILD_PUBLIC_INC_DIR} )
  file( MAKE_DIRECTORY ${EMBEDDED_BUILD_PRIVATE_INC_DIR} )

  add_custom_target(copy_codegen_files ALL
                    COMMENT "Copying source files needed for code generation" )

  # Copy a sample makefile into the build tree
  add_custom_command(OUTPUT "${EMBEDDED_BUILD_ROOT_DIR}/Makefile"
                     COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/configure/Makefile.emosqp" "${EMBEDDED_BUILD_ROOT_DIR}/Makefile"
                     DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/configure/Makefile.emosqp"
                     COMMENT "Copying codegen makefile" )

  add_custom_target( copy_codegen_makefile DEPENDS "${EMBEDDED_BUILD_ROOT_DIR}/Makefile" )
  add_dependencies( copy_codegen_files copy_codegen_makefile )
endif()

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

get_property(
  osqplib_link_libs
  TARGET OSQPLIB
  PROPERTY LINK_LIBRARIES)

# ----------------------------------------------
# Language-specific compilation checks
# ----------------------------------------------
if(PYTHON)
  if(NOT PYTHON_INCLUDE_DIRS)
    message(FATAL_ERROR "You need Python include directories to build the Python interface")
  endif(NOT PYTHON_INCLUDE_DIRS)

  include_directories(${PYTHON_INCLUDE_DIRS})
  add_definitions(-DPYTHON)

  if(OSQP_BUILD_UNITTESTS)
    message(STATUS "Disabling OSQP_BUILD_UNITTESTS because we are building Python interface")
    set(OSQP_BUILD_UNITTESTS OFF)
  endif(OSQP_BUILD_UNITTESTS)
endif(PYTHON)

if(MATLAB)
  find_package(Matlab)
  if(NOT Matlab_FOUND)
    message(FATAL_ERROR "You need Matlab libraries to build the Matlab interface")
  endif(NOT Matlab_FOUND)
  message(STATUS "Matlab root is " ${Matlab_ROOT_DIR})

  include_directories(${Matlab_INCLUDE_DIRS})
  add_definitions(-DMATLAB)

  # Insist on the pre 2018 complex data API so that mxGetPr will work correctly
  add_definitions(-DMATLAB_MEXSRC_RELEASE=R2017b)
  message(STATUS "Using Matlab pre-2018a API for mxGetPr compatibility")

  if(OSQP_BUILD_UNITTESTS)
    message(STATUS "Disabling OSQP_BUILD_UNITTESTS because we are building Matlab interface")
    set(OSQP_BUILD_UNITTESTS OFF)
  endif(OSQP_BUILD_UNITTESTS)
endif(MATLAB)

if(R_LANG)
  find_package(R)
  if(NOT R_FOUND)
    message(FATAL_ERROR "You need R libraries to build the R interface")
  endif(NOT R_FOUND)

  message(STATUS "R exec is: " ${R_EXEC})
  message(STATUS "R root dir is: " ${R_ROOT_DIR})
  message(STATUS "R includes are in: " ${R_INCLUDE_DIRS})

  include_directories(${R_INCLUDE_DIRS})
  add_definitions(-DR_LANG)

  if(OSQP_BUILD_UNITTESTS)
    message(STATUS "Disabling OSQP_BUILD_UNITTESTS because we are building the R interface")
    set(OSQP_BUILD_UNITTESTS OFF)
  endif(OSQP_BUILD_UNITTESTS)
endif(R_LANG)

if(PYTHON OR MATLAB OR R_LANG)
  # Disable shared library and demo exe when building the interfaces
  set(OSQP_BUILD_SHARED_LIB OFF)
  set(OSQP_BUILD_DEMO_EXE OFF)
endif()

# ----------------------------------------------
# Static Library - osqpstatic
# ----------------------------------------------
message( STATUS "Build static library: " ${OSQP_BUILD_STATIC_LIB} )

if( OSQP_BUILD_STATIC_LIB )
  add_library(osqpstatic STATIC
              $<TARGET_OBJECTS:OSQPLIB>
              "${CMAKE_CURRENT_SOURCE_DIR}/src/osqp_api.c")

  # Append the algebra name to the library file if desired
  if(OSQP_LIB_SUFFIX)
    set_target_properties(osqpstatic PROPERTIES OUTPUT_NAME "osqpstatic_${OSQP_LIB_SUFFIX}")
  else()
    set_target_properties(osqpstatic PROPERTIES OUTPUT_NAME "osqpstatic")
  endif()


  if(OSQP_ALGEBRA_BUILTIN)
    # Transitive dependency on OBJECT library does not work. See https://gitlab.kitware.com/cmake/cmake/-/issues/18682
    target_sources(osqpstatic PRIVATE $<TARGET_OBJECTS:qdldlobject>)
  endif()
  target_include_directories(osqpstatic PRIVATE ${osqplib_includes})
  target_include_directories(
    osqpstatic PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/public;${CMAKE_CURRENT_BINARY_DIR}/include/public>"
                      "$<INSTALL_INTERFACE:$<INSTALL_PREFIX>/${CMAKE_INSTALL_INCLUDEDIR}/osqp>")

  if(MATLAB)
    target_link_libraries(osqpstatic ${Matlab_LIBRARIES})
  endif(MATLAB)
endif()

# ----------------------------------------------
# Shared Library - osqp
# ----------------------------------------------
message( STATUS "Build shared library: " ${OSQP_BUILD_SHARED_LIB} )

if(OSQP_BUILD_SHARED_LIB)
  add_library(osqp SHARED
              $<TARGET_OBJECTS:OSQPLIB>
              "${CMAKE_CURRENT_SOURCE_DIR}/src/osqp_api.c")
  if(OSQP_ALGEBRA_BUILTIN)
    # Transitive dependency on OBJECT library does not work. See https://gitlab.kitware.com/cmake/cmake/-/issues/18682
    target_sources(osqp PRIVATE $<TARGET_OBJECTS:qdldlobject>)
  endif()
  target_include_directories(osqp PRIVATE ${osqplib_includes})
  target_include_directories(osqp PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
                                         "$<INSTALL_INTERFACE:$<INSTALL_PREFIX>/${CMAKE_INSTALL_INCLUDEDIR}/osqp>")

  # Link against the libraries for the algebras
  target_link_libraries(osqp ${osqplib_link_libs})

  # Declare that we are building the shared library to get proper symbol exports.
  # Shared library consumers should also define OSQP_SHARED_LIB to get the library
  # exports properly, so we do it for them in the CMake interface by defining it as
  # a PUBLIC compile definition.
  target_compile_definitions(osqp PRIVATE BUILDING_OSQP)
  target_compile_definitions(osqp PUBLIC  OSQP_SHARED_LIB)

  # Modify the soname of the library to include the algebra if desired
  if(OSQP_LIB_SUFFIX)
    set_target_properties(osqp PROPERTIES LIBRARY_OUTPUT_NAME "osqp_${OSQP_LIB_SUFFIX}")
  endif()
endif()

# ----------------------------------------------
# Application - osqp_demo
# ----------------------------------------------
message( STATUS "Build demo executable: " ${OSQP_BUILD_DEMO_EXE} )

if(OSQP_BUILD_DEMO_EXE)
  add_executable(osqp_demo ${PROJECT_SOURCE_DIR}/examples/osqp_demo.c)
  #target_include_directories(osqp_demo PRIVATE ${osqplib_includes})
  target_link_libraries(osqp_demo osqpstatic ${osqplib_link_libs})

  if(OSQP_CODEGEN)
    add_executable(osqp_codegen_demo ${PROJECT_SOURCE_DIR}/examples/osqp_codegen_demo.c)
    target_link_libraries(osqp_codegen_demo osqpstatic)
  endif()
endif()

# ----------------------------------------------
# Installation / Uninstallation
# ----------------------------------------------
install(FILES ${osqp_headers} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/osqp")

if(OSQP_BUILD_STATIC_LIB)
  install(
    TARGETS osqpstatic
    EXPORT osqpstatic
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")

  export(
    EXPORT osqpstatic
    FILE "${CMAKE_CURRENT_BINARY_DIR}/osqpstatic-targets.cmake"
    NAMESPACE osqp::)

  install(
    EXPORT osqpstatic
    FILE osqpstatic-targets.cmake
    NAMESPACE osqp::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/osqp")
endif()

if(OSQP_BUILD_SHARED_LIB)
  install(
    TARGETS osqp
    EXPORT osqp
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")

  export(
    EXPORT osqp
    FILE "${CMAKE_CURRENT_BINARY_DIR}/osqp-targets.cmake"
    NAMESPACE osqp::)

  install(
    EXPORT osqp
    FILE osqp-targets.cmake
    NAMESPACE osqp::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/osqp")
endif()

if(OSQP_BUILD_STATIC_LIB OR OSQP_BUILD_SHARED_LIB)
  # generate the config file that is includes the exports
  configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/configure/osqp-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/osqp-config.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/osqp"
    NO_SET_AND_CHECK_MACRO
    NO_CHECK_REQUIRED_COMPONENTS_MACRO)

  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/osqp-config-version.cmake"
    VERSION "${OSQP_VERSION}"
    COMPATIBILITY SameMajorVersion)

  install(FILES
          "${CMAKE_CURRENT_BINARY_DIR}/osqp-config.cmake"
          "${CMAKE_CURRENT_BINARY_DIR}/osqp-config-version.cmake"
          DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/osqp)
endif()

if(OSQP_CODEGEN)
  install(DIRECTORY
          "${EMBEDDED_BUILD_ROOT_DIR}/"
          DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/osqp/codegen_files")
endif()

if(NOT TARGET uninstall)
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/configure/cmake/cmake_uninstall.cmake.in"
                 "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake" IMMEDIATE @ONLY)

  add_custom_target(uninstall COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
endif()

# ----------------------------------------------
# Testing
# ----------------------------------------------
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND OSQP_BUILD_UNITTESTS)
  include(CTest)

  message( STATUS "Building unit tests" )

  if( OSQP_COVERAGE_CHECK )
    message( STATUS "Enabling code coverage file generation" )
  endif()

  add_subdirectory(tests)
endif()
