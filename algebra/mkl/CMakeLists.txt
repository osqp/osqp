if(OSQP_USE_LONG)
  set(MKL_INTERFACE "ilp64")
else()
  set(MKL_INTERFACE "lp64")
endif()

# Link against the single dynamic library version of the MKL library
set(MKL_LINK "sdl")

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
