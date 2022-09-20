file(
  GLOB
  SRC_FILES
  CONFIGURE_DEPENDS
  *.h
  *.cu
  src/*.cu
  lin_sys/indirect/*.h
  lin_sys/indirect/*.cu)

target_sources(OSQPLIB PRIVATE ${SRC_FILES})

target_include_directories(OSQPLIB PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} include lin_sys/indirect)

option(OSQP_CUDA_STATIC_LINKAGE "Statically link all CUDA libraries" OFF)
mark_as_advanced(OSQP_CUDA_STATIC_LINKAGE)

if(OSQP_CUDA_STATIC_LINKAGE)
  message(STATUS "Statically linking CUDA libraries")

  # cublasLt_static is a dependency of cublas_static.
  target_link_libraries(OSQPLIB cublas_static cusparse_static cublasLt_static)
else()
  target_link_libraries(OSQPLIB cublas cusparse)
endif()
