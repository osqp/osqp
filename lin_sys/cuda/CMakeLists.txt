# Add CUDA linear systems solvers


# CUDA-PCG (Default)
# -------------------------
add_subdirectory(cuda_pcg)

set(cuda_linsys_solvers $<TARGET_OBJECTS:linsys_cuda_pcg>)

set(cuda_linsys_solvers_includes ${cuda_pcg_includes})

# Pass upwards
set(cuda_linsys_solvers ${cuda_linsys_solvers} PARENT_SCOPE)
set(cuda_linsys_solvers_includes ${cuda_linsys_solvers_includes} PARENT_SCOPE)
