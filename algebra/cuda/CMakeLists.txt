file(
  GLOB
  SRC_FILES
  CONFIGURE_DEPENDS
  *.h
  *.cu
  src/*.cu
  lin_sys/indirect/*.h
  lin_sys/indirect/*.cu)

# Search for cuDSS (Optional dependency, might not be always available)
find_package(cudss)

if(${cudss_FOUND})
  message(STATUS "Enabling cuDSS solver")
  target_compile_definitions(OSQPLIB PUBLIC CUDSS_AVAILABLE=1)
else()
  message(STATUS "Not enabling cuDSS solver - cuDSS not found")
endif()

option(OSQP_CUDA_STATIC_LINKING "Statically link all CUDA libraries (WARNING: This will greatly increase the file size of the library)" OFF)
mark_as_advanced(OSQP_CUDA_STATIC_LINKING)


target_sources(OSQPLIB PRIVATE
               ${SRC_FILES}
               "$<$<BOOL:${cudss_FOUND}>:${CMAKE_CURRENT_SOURCE_DIR}/lin_sys/direct/cuda_cudss.cu>"
               "$<$<BOOL:${cudss_FOUND}>:${OSQP_ALGEBRA_ROOT}/_common/kkt.c>"
               "$<$<BOOL:${cudss_FOUND}>:${OSQP_ALGEBRA_ROOT}/_common/csc_utils.c>"
               "$<$<BOOL:${cudss_FOUND}>:${OSQP_ALGEBRA_ROOT}/_common/csc_math.c>")

target_include_directories(OSQPLIB PRIVATE
                           ${CMAKE_CURRENT_SOURCE_DIR}
                           include
                           lin_sys/indirect
                           "$<$<BOOL:${cudss_FOUND}>:${cudss_INCLUDE_DIR}>"
                           "$<$<BOOL:${cudss_FOUND}>:${CMAKE_CURRENT_SOURCE_DIR}/lin_sys/direct>"
                           "$<$<BOOL:${cudss_FOUND}>:${OSQP_ALGEBRA_ROOT}/_common>")


if(OSQP_CUDA_STATIC_LINKING)
  message(STATUS "Statically linking CUDA libraries")

  # cublasLt_static is a dependency of cublas_static.
  target_link_libraries(OSQPLIB
                        CUDA::cudart
                        CUDA::cublas_static
                        CUDA::cusparse_static
                        CUDA::cublasLt_static
                        "$<$<BOOL:${cudss_FOUND}>:cudss_static>")
else()
  target_link_libraries(OSQPLIB
                        CUDA::cudart
                        CUDA::cublas
                        CUDA::cusparse
                        "$<$<BOOL:${cudss_FOUND}>:cudss>")
endif()

configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/findAlgebraDependency.cmake.in
                ${CMAKE_CURRENT_BINARY_DIR}/../../osqp-findAlgebraDependency.cmake
                @ONLY )

