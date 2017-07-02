# Add direct linear systems solvers

# suitesparse
add_subdirectory(suitesparse)

# suitesparse includes
include_directories(${suitesparse_ldl_includes})

# create library
add_library(linsys_suitesparse_ldl OBJECT ${suitesparse_ldl_sources})
