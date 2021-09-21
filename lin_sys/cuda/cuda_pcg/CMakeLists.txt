# Add CUDA-PCG

set(cuda_pcg_includes
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set(cuda_pcg_sources
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_pcg_interface.h
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_pcg_interface.c
    ${CMAKE_CURRENT_SOURCE_DIR}/include/cuda_pcg_constants.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/cuda_pcg.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda_pcg.cu
)

# Create object library for linear system solver interface
add_library(linsys_cuda_pcg OBJECT ${cuda_pcg_sources})

target_include_directories(linsys_cuda_pcg PRIVATE
                           ${cuda_pcg_includes}
                           ${osqp_api_includes}
                           ${algebra_impl_includes})

# Pass upwards
set(cuda_pcg_includes ${cuda_pcg_includes} PARENT_SCOPE)
