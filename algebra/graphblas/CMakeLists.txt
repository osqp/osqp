# Use the QDLDL solver in this algebra
include(${OSQP_ALGEBRA_ROOT}/_common/lin_sys/direct/qdldl.cmake)

find_package(GraphBLAS REQUIRED)

target_sources(
  OSQPLIB
  PRIVATE ../_common/csc_math.h
          ../_common/csc_math.c
          algebra_impl.h
          algebra_libs.c
          custom_operators.c
          vector.c
          matrix.c
          ${NON_EMBEDDED_SRC_FILES}
          ${LIN_SYS_QDLDL_SRC_FILES}
          $<TARGET_OBJECTS:qdldlobject> )

target_include_directories(
  OSQPLIB
  PRIVATE ../_common
          ${CMAKE_CURRENT_SOURCE_DIR}
          ${LIN_SYS_QDLDL_INC_PATHS}
          ${GraphBLAS_INCLUDE_DIRS} )

target_link_libraries(
  OSQPLIB
  GraphBLAS_LIBRARIES )
