# Define the algebra implementation

set(algebra_impl_src
    ${CMAKE_CURRENT_SOURCE_DIR}/algebra_libs.c
    ${CMAKE_CURRENT_SOURCE_DIR}/matrix.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/vector.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda_csr.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda_handler.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda_lin_alg.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda_malloc.cu
)

set(algebra_impl_headers
    ${CMAKE_CURRENT_SOURCE_DIR}/algebra_types.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/csr_type.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/cuda_configure.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/cuda_csr.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/cuda_handler.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/cuda_lin_alg.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/cuda_malloc.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/cuda_wrapper.h
)

if (IS_WINDOWS)
    set(helper_cuda_header_dir $ENV{NVCUDASAMPLES_ROOT}/common/inc)
else ()
    set(helper_cuda_header_dir ${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc)
endif()

set(algebra_impl_includes
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${helper_cuda_header_dir}
)

set(algebra_impl ${algebra_impl_src} ${algebra_impl_headers} PARENT_SCOPE)
set(algebra_impl_includes ${algebra_impl_includes} PARENT_SCOPE)

