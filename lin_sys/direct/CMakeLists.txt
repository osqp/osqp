# Add direct linear systems solvers

# suitesparse
add_subdirectory(suitesparse)

# suitesparse includes
include_directories(${suitesparse_includes})

# create library
add_library(linsys_direct OBJECT ${suitesparse_sources})
