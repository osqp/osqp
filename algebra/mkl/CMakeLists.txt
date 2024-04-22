if(OSQP_USE_LONG)
  set(MKL_INTERFACE "ilp64")
else()
  set(MKL_INTERFACE "lp64")
endif()

# Select the MKL library to link against.
# Link against the single dynamic library version of the MKL library by default.
set(MKL_LINK "sdl" CACHE STRING "MKL linkage")

if(${MKL_LINK} STREQUAL "sdl")
  target_compile_definitions(OSQPLIB PUBLIC MKL_LINK_SDL=1)
endif()

# MKL CMake setup
# ----------------------------------------------
# If MKL_ROOT is not explicitly set, we need to figure out where to search for MKL.
#
# If the environment tells us where MKL is (using the MKLROOT variable set by the MKL
# variable scripts), use that as the CMake search path.
#
# If the environment doesn't tell us where to find MKL and a Conda environment exists,
# try to find MKL in Conda (it may have been installed using `conda install -c intel mkl-devel`).
# ----------------------------------------------
if( NOT MKL_ROOT )
  if( DEFINED ENV{MKLROOT} )
    # MKL's setup wrapper script doesn't properly add the MKL CMake files to the path
    # so do that here instead.
    list( PREPEND CMAKE_PREFIX_PATH $ENV{MKLROOT} )
  elseif( DEFINED ENV{CONDA_PREFIX} )
    message( STATUS "Detected Conda environment - adding Conda to the CMake path" )
    list( PREPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}" )
  endif()
endif()

find_package(MKL CONFIG REQUIRED)

target_sources(
  OSQPLIB
  PRIVATE ../_common/csc_math.h
          ../_common/csc_math.c
          ../_common/csc_utils.h
          ../_common/csc_utils.c
          ../_common/kkt.h
          ../_common/kkt.c
          ../_common/reduced_kkt.h
          ../_common/reduced_kkt.c
          vector.c
          matrix.c
          algebra_impl.h
          algebra_libs.c
          lin_sys/direct/pardiso_interface.h
          lin_sys/direct/pardiso_interface.c
          lin_sys/indirect/mkl-cg_interface.h
          lin_sys/indirect/mkl-cg_interface.c)

target_include_directories(OSQPLIB PRIVATE
                           ../_common
                           ${CMAKE_CURRENT_SOURCE_DIR}
                           ${CMAKE_CURRENT_SOURCE_DIR}/lin_sys/direct
                           ${CMAKE_CURRENT_SOURCE_DIR}/lin_sys/indirect)

target_include_directories(OSQPLIB PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
target_compile_options(OSQPLIB PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)

target_link_libraries(OSQPLIB MKL::MKL)

configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/findAlgebraDependency.cmake.in
                ${CMAKE_CURRENT_BINARY_DIR}/../../osqp-findAlgebraDependency.cmake
                @ONLY )
