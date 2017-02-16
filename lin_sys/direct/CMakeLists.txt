# Add direct linear systems solvers

# suitesparse
add_subdirectory(suitesparse)

# create library
add_library(linsys_direct ${suitesparse_source_files})
